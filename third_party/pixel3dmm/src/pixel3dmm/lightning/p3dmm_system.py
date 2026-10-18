from PIL import Image, ImageDraw
import os
import torch
import numpy as np
import pytorch_lightning as L
import torch.nn as nn

from pixel3dmm.lightning.utils import CosineWarmupScheduler, WarmupScheduler
from pixel3dmm.lightning.p3dmm_network import Network
from pixel3dmm import env_paths


def fov_to_ixt(fov, reso=512):
    ixt = torch.eye(3).float().unsqueeze(0).repeat(fov.shape[0], 1, 1).to(fov.device)
    ixt[:, 0, 2] = reso / 2
    ixt[:, 1, 2] = reso / 2
    focal = .5 * reso / torch.tan(.5 * fov)
    ixt[:, 0, 0] = focal
    ixt[:, 1, 1] = focal
    return ixt


def batch_rodrigues(
    rot_vecs: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def pad_to_3_channels(img):
    if img.shape[-1] == 3:
        return img
    elif img.shape[-1] == 1:
        return np.concatenate([img, np.zeros_like(img[..., :1]), np.zeros_like(img[..., :1])], axis=-1)
    elif img.shape[-1] == 2:
        return np.concatenate([img, np.zeros_like(img[..., :1])], axis=-1)
    else:
        raise ValueError('too many dimensions in prediction type!')


class system(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.glctx = None
        self.cfg = cfg
        self.net = Network(cfg)

        vertex_weight_mask = np.load(f'{env_paths.VERTEX_WEIGHT_MASK}')

        self.register_buffer('vertex_weight_mask', torch.from_numpy(vertex_weight_mask).float())



        self.validation_step_outputs = []
        self.validation_step_outputs_per_dataset = []

        self.dataset_types = [
            'facescape',
            'nphm',
            'ava',
        ]


        self.do_eval = True

        self.alpha = 1.0

        self.save_hyperparameters()

        self.loss_weights = {
            'albedo': 1.0,  # 1.0/0.13,
            'depth': 1.0,
            'pos_map': 1.0,  # 1.0/0.0006,
            'pos_map_can': 1.0,  # 1.0/0.0006,
            'normals': 0.1,  # TODO achtung #1.0/0.03,
            'normals_can': 1.0,  # 1.0/0.03,
            'uv_map': 10.0,  # 1.0/0.001,
            'nocs': 1.0,  # 1.0/0.0006,
        }


    def training_step(self, batch, batch_idx):


        output, conf = self.net(batch)

        B = output[list(output.keys())[0]].shape[0]
        V = output[list(output.keys())[0]].shape[1]

        c_map = None



        losses = {}


        if 'normals' in self.cfg.model.prediction_type:

            gt_normals = batch['normals'].permute(0, 1, 4, 2, 3)
            if conf is None:
                losses['normals'] = (batch['tar_msk'].unsqueeze(2) * (gt_normals - output['normals'])).abs().mean()
            else:
                losses['normals'] = (batch['tar_msk'].unsqueeze(2) * (
                        c_map * (gt_normals - output['normals']) - self.alpha * torch.log(c_map))).abs().mean()

            if self.cfg.model.pred_disentangled:
                gt_normals_can = batch['normals_can'].permute(0, 1, 4, 2, 3)
                if conf is None:
                    losses['normals_can'] = (
                            batch['tar_msk'].unsqueeze(2) * (gt_normals_can - output['normals_can'])).abs().mean()
                else:
                    losses['normals_can'] = (batch['tar_msk'].unsqueeze(2) * (
                        c_map * (gt_normals_can - output['normals_can']) - self.alpha * torch.log(
                        c_map))).abs().mean()


        for prediction_type in ['uv_map',  'depth', 'nocs']:
            if prediction_type in self.cfg.model.prediction_type:
                weight_mask = torch.ones_like(output[prediction_type])
                if prediction_type == 'uv_map' or (prediction_type == 'nocs'):  # ATTENTION: only for nocs?
                    weight_mask = batch['uv_masks'].unsqueeze(2).float() + 0.2
                gt_pos_map = batch[prediction_type].permute(0, 1, 4, 2, 3)
                if conf is None:
                    losses[prediction_type] = (weight_mask * batch['tar_msk'].unsqueeze(2) * (
                            gt_pos_map - output[prediction_type])).abs().mean()
                else:
                    losses[prediction_type] = (weight_mask * batch['tar_msk'].unsqueeze(2) * (
                            c_map * (gt_pos_map - output[prediction_type]) - self.alpha * torch.log(
                            c_map))).abs().mean()

        total_loss = 0

        loss = 0
        for k in losses.keys():
            if k in self.loss_weights:
                loss += self.loss_weights[k] * losses[k]
            else:
                loss += losses[k]



        self.log(f'train/loss', loss.item(), prog_bar=False)
        # for prediction_type in self.cfg.model.prediction_type:
        for k in losses.keys():
            if k in self.cfg.model.prediction_type:
                self.log(f'train/loss_{k}', losses[k])
        if self.cfg.model.pred_disentangled:
            for k in losses.keys():
                if k[:-4] in self.cfg.model.prediction_type:
                    self.log(f'train/loss_{k}', losses[k])


        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

        do_vis = 0 == self.trainer.global_step % 3000
        if do_vis and (self.trainer.local_rank == 0):
            output, conf = self.net(batch)


            self.vis_results({k: v.detach() for (k, v) in output.items()}, conf, batch, prex='train')
            self.do_eval = True
            torch.cuda.empty_cache()


        return loss



    def optimizer_step(
        self,
        *args, **kwargs
    ):
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        grads = [
            param.grad.detach().flatten()
            for param in self.parameters()
            if param.grad is not None
        ]
        if len(grads) > 0:
            norm = torch.cat(grads).norm()
            self.log(f'grad/norm', norm.item(), prog_bar=False)  # , sync_dist=True)

            if (norm > 10000 and self.global_step > 20 or torch.isnan(norm)):
                valid_gradients = False

            if not valid_gradients:
                print(
                    f'detected inf or nan values in gradients. not updating model parameters, OTHER FUNCTION threshold: {10000}, value: {norm.item()}')
                self.zero_grad()
                for param in self.parameters():
                    param.grad = None

        L.LightningModule.optimizer_step(self, *args, **kwargs)


    def validation_step(self, batch, batch_idx):


        self.net.eval()
        output, conf = self.net(batch)

        B = output[list(output.keys())[0]].shape[0]
        V = output[list(output.keys())[0]].shape[1]



        loss_dict = {}

        dataset_indices = {}



        val_losses = {}
        for prediction_type in ['uv_map', 'depth', 'nocs']:
            if prediction_type in self.cfg.model.prediction_type:
                gt_pos_map = batch[prediction_type].permute(0, 1, 4, 2, 3)
                weight_mask = torch.ones_like(output[prediction_type])
                if prediction_type == 'uv_map' or (prediction_type == 'nocs'):  # ATTENTION: only for nocs?
                    weight_mask = batch['uv_masks'].unsqueeze(2).float() + 0.2

                val_losses[prediction_type] = (weight_mask * batch['tar_msk'].unsqueeze(2) * (
                        gt_pos_map - output[prediction_type])).abs().mean()
                loss_dict[f'loss/{prediction_type}'] = val_losses[prediction_type].item()

        if 'normals' in self.cfg.model.prediction_type:
            prediction_type = 'normals'
            gt_pos_map = batch[prediction_type].permute(0, 1, 4, 2, 3)

            val_losses[prediction_type] = (
                batch['tar_msk'].unsqueeze(2) * (gt_pos_map - output[prediction_type])).abs().mean()

            loss_dict[f'loss/{prediction_type}'] = val_losses[prediction_type].item()

            if self.cfg.model.pred_disentangled:
                prediction_type = 'normals_can'
                gt_pos_map = batch[prediction_type].permute(0, 1, 4, 2, 3)

                val_losses[prediction_type] = (
                    batch['tar_msk'].unsqueeze(2) * (gt_pos_map - output[prediction_type])).abs().mean()

                loss_dict[f'loss/{prediction_type}'] = val_losses[prediction_type].item()

        # if self.cfg.model.prediction_type == 'depth_si':
        #    loss, pred_scale, target_scale = simae2_loss(output, batch['depth'].permute(0, 1, 4, 2, 3), batch['tar_msk'].unsqueeze(2), c_map=c_map, alpha=self.alpha)
        #    self.validation_step_outputs.append({'loss': loss.item()})

        val_loss = 0

        for prediction_type in self.cfg.model.prediction_type:
            val_loss += self.loss_weights[prediction_type] * val_losses[prediction_type]


        loss_dict['loss/total'] = val_loss.item()
        self.validation_step_outputs.append(loss_dict)

        #print('GLOBAL_STEP:', self.trainer.global_step)
        if self.do_eval and self.trainer.local_rank == 0:
            output, conf = self.net(batch)
            if conf is not None:
                conf = conf.detach()
            tmp_dict = {k: v.detach() for (k, v) in output.items()}
            self.vis_results(tmp_dict, conf, batch, prex='val')
            self.do_eval = False
            torch.cuda.empty_cache()

        return val_loss

    def on_validation_epoch_end(self):
        # for key in keys:
        #    prog_bar = True if key in ['psnr','mask','depth'] else False
        metric_mean = np.stack([np.array(x['loss/total']) for x in self.validation_step_outputs]).mean()
        self.log(f'val/loss', metric_mean, prog_bar=False, sync_dist=True)
        if self.net.n_facial_components == 0:

            for prediction_type in self.cfg.model.prediction_type:
                metric_mean_pred_type = np.stack(
                    [np.array(x[f'loss/{prediction_type}']) for x in self.validation_step_outputs]).mean()
                self.log(f'val/loss_{prediction_type}', metric_mean_pred_type, sync_dist=True)

        for dataset_type in self.dataset_types:
            for loss_type in self.validation_step_outputs[0].keys():
                content = [np.array(x[dataset_type][loss_type]) for x in self.validation_step_outputs_per_dataset if loss_type in x[dataset_type]]
                if len(content) > 0:
                    metric_mean = np.nanmean(np.stack(content))
                    self.log(f'val_{dataset_type}/{loss_type}', metric_mean, sync_dist=True)

        self.validation_step_outputs.clear()  # free memory
        torch.cuda.empty_cache()

    def vis_results(self, output, conf, batch, prex):
        out_folder = f'{self.cfg.reconstruction_folder}/{prex}_{self.trainer.global_step}/'
        os.makedirs(out_folder, exist_ok=True)
        output_gpu = {k: v for k, v in output.items()}
        output = {k: v.cpu() for k, v in output.items()}
        if self.net.n_facial_components == 0:
            output_rows = {}

            for predictiont_type in ['normals', 'albedo', 'uv_map', 'nocs']:
                if predictiont_type in self.cfg.model.prediction_type:
                    output_rows[predictiont_type] = (batch['tar_msk'][..., None].float() * batch[predictiont_type]).permute(0, 1, 4, 2, 3).detach().cpu()
                if predictiont_type in self.cfg.model.prediction_type and predictiont_type == 'normals' and self.cfg.model.pred_disentangled:
                    output_rows['normals_can'] = (batch['tar_msk'][..., None].float() * batch['normals_can']).permute(0, 1, 4, 2, 3).detach().cpu()

            gt_rgb = batch['tar_rgb'].permute(0, 1, 4, 2, 3).detach().cpu()


            for i_batch in range(output_rows[self.cfg.model.prediction_type[0]].shape[0]):

                modalities = []
                prediction_types = self.cfg.model.prediction_type.copy()  # ['pos_map', 'normals', 'albedo', 'uv_map']
                if self.cfg.model.pred_disentangled and "pos_map" in prediction_types:
                    prediction_types.append('pos_map_can')
                if self.cfg.model.pred_disentangled and "normals" in prediction_types:
                    prediction_types.append('normals_can')
                if self.cfg.model.pred_disentangled and "uv_map" in prediction_types:
                    prediction_types.append('disps')

                for prediction_type in prediction_types:
                    rows = []
                    for i_view in range(output_rows[prediction_type].shape[1]):
                        with torch.no_grad():
                            mini = min(output_rows[prediction_type][i_batch, i_view].min().item(),
                                       output[prediction_type][i_batch, i_view].min().item())
                            tmp_gt_pos_map = output_rows[prediction_type][i_batch, i_view].clone() - mini
                            tmp_output = output[prediction_type][i_batch, i_view].clone() - mini
                            maxi = max(tmp_gt_pos_map.max().item(), tmp_output.max().item())
                            tmp_gt_pos_map = tmp_gt_pos_map / maxi
                            tmp_output = tmp_output / maxi

                            catted = [
                                gt_rgb[i_batch, i_view].permute(1, 2, 0).detach().cpu().numpy(),
                                pad_to_3_channels(
                                    (batch['tar_msk'][i_batch, i_view].cpu() * tmp_gt_pos_map.cpu()).permute(1, 2,
                                                                                                             0).detach().cpu().numpy()),
                                pad_to_3_channels(tmp_output.permute(1, 2, 0).detach().cpu().float().numpy()),
                            ]

                            if conf is not None:
                                mini_conf = conf[i_batch, i_view].min()
                                tmp_conf = conf[i_batch, i_view].clone() - mini_conf
                                maxi_conf = tmp_conf.max()
                                tmp_conf = tmp_conf / maxi_conf
                                catted.append(
                                    pad_to_3_channels(tmp_conf.permute(1, 2, 0).detach().cpu().float().numpy()))

                            catted = (np.concatenate(catted, axis=1) * 255).astype(np.uint8)

                            rows.append(catted)
                    modalities.append(np.concatenate(rows, axis=0))

                catted = Image.fromarray(np.concatenate(modalities, axis=0))
                scene_name = batch['meta']['scene'][i_batch]
                catted.save(f'{out_folder}/{scene_name}.png')  # , quality=90)




        keys = list(output.keys())
        for k in keys:
            del output[k]
        del output
        del gt_rgb
        keys = list(output_rows.keys())
        for k in keys:
            del output_rows[k]
        del output_rows

        torch.cuda.empty_cache()
        # pll.show()

    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size * self.trainer.max_epochs * self.cfg.train.limit_train_batches // (
                self.trainer.accumulate_grad_batches * num_devices)
        return int(num_steps)

    def configure_optimizers(self):
        decay_params, no_decay_params = [], []

        invalid_params = []
        all_backbone_params = []
        all_non_backbone_params = []
        backbone_params = []
        backbone_params_no_decay = []
        # add all bias and LayerNorm params to no_decay_params
        for name, module in self.named_modules():
            if name == 'flame' or name == 'flame_generic':
                invalid_params.extend([p for p in module.parameters()])
            else:
                if isinstance(module, nn.LayerNorm):
                    if 'img_encoder' in name:
                        backbone_params_no_decay.extend([p for p in module.parameters()])
                    else:
                        no_decay_params.extend([p for p in module.parameters()])
                elif hasattr(module, 'bias') and module.bias is not None:
                    if 'img_encoder' in name:
                        backbone_params_no_decay.append(module.bias)
                    else:
                        no_decay_params.append(module.bias)

                if 'img_encoder' in name:
                    all_backbone_params.extend([p for p in module.parameters()])
                else:
                    all_non_backbone_params.extend([p for p in module.parameters()])

        # add remaining parameters to decay_params
        _no_decay_ids = set(map(id, no_decay_params))
        _all_backbone_ids = set(map(id, all_backbone_params))
        _all_non_backbone_ids = set(map(id, all_non_backbone_params))
        _backbone_no_decay_ids = set(map(id, backbone_params_no_decay))
        _invalid_ids = set(map(id, invalid_params))
        decay_params = [p for p in self.parameters() if
                        id(p) not in _no_decay_ids and id(p) not in _all_backbone_ids and id(p) not in _invalid_ids]
        decay_params_backbone = [p for p in self.parameters() if
                                 id(p) not in _backbone_no_decay_ids and id(p) not in _all_non_backbone_ids and id(
                                     p) not in _invalid_ids]
        no_decay_params = [p for p in no_decay_params if id(p) not in _invalid_ids]
        no_decay_params_backbone = [p for p in backbone_params_no_decay if id(p) not in _invalid_ids]

        # filter out parameters with no grad
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))
        decay_params_backbone = list(filter(lambda p: p.requires_grad, decay_params_backbone))
        no_decay_params_backbone = list(filter(lambda p: p.requires_grad, no_decay_params_backbone))

        # Optimizer
        opt_groups = [
            {'params': decay_params, 'weight_decay': self.cfg.train.weight_decay, 'lr': self.cfg.train.lr},
            {'params': decay_params_backbone, 'weight_decay': self.cfg.train.weight_decay,
             'lr': self.cfg.train.lr_backbone},
            {'params': no_decay_params, 'weight_decay': 0.0, 'lr': self.cfg.train.lr},
            {'params': no_decay_params_backbone, 'weight_decay': 0.0, 'lr': self.cfg.train.lr_backbone},
        ]
        optimizer = torch.optim.AdamW(
            opt_groups,
            betas=(self.cfg.train.beta1, self.cfg.train.beta2),
        )

        total_global_batches = self.num_steps()

        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_iters=self.cfg.train.warmup_iters,
            max_iters=total_global_batches,
        )

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    'scheduler': scheduler,
                    'interval': 'step'  # or 'epoch' for epoch-level updates
                }}
