import torch
import os
import numpy as np
from pytorch3d.ops import knn_points
from pixel3dmm import env_paths


def get_albedo_loss(gt, pred, mask):
    gt_albedo = gt[:, :3, :, :].permute(0, 2, 3, 1)
    albedo_loss = ((gt_albedo - pred.permute(0, 2, 3, 1)) * mask[:, 0,
                                                                            ...].unsqueeze(
        -1)).abs().mean()
    return albedo_loss

def get_pos_map_loss(gt, pred, mask):
    gt_pos_map = gt.permute(0, 2, 3, 1)
    tmp = pred
    tmp *= 4
    tmp = torch.stack([-tmp[:, 0, ...], tmp[:, 2, ...], tmp[:, 1, ...]], dim=1)
    tmp /= 1.25

    tmp[:, 1] += 0.2
    l_map = (gt_pos_map -
                     tmp.permute(0, 2, 3, 1))
    valid = l_map < 0.015
    pos_map_loss = (l_map * valid.float() * mask).abs().mean()
    return pos_map_loss


def get_pos_map_loss_corresp(gt, pred, omit_mean=False):
    tmp = pred
    tmp *= 4
    tmp = torch.stack([-tmp[:, 0], tmp[:, 2], tmp[:, 1]], dim=1)
    tmp /= 1.25

    tmp[:, 1] += 0.2
    outliers = (gt - tmp).abs().sum(dim=-1) > 0.066
    if omit_mean:
        pos_map_loss = (gt - tmp) * (~outliers).float().unsqueeze(-1)
        #pos_map_loss = gt - tmp
    else:
        pos_map_loss = ((gt - tmp)[~outliers, :]).abs().mean()
    return pos_map_loss


class UVLoss():

    def __init__(self, stricter_mask : bool = False, delta_uv=0.00005, delta_nocs=0.0001, dist_uv=15):
        self.delta = delta_uv
        self.delta_nocs = delta_nocs
        self.dist_uv = dist_uv
        self.valid_verts = None
        self.valid_verts_nocs = None
        self.stricter_mask = stricter_mask

        if self.stricter_mask:
            self.valid_verts = np.load(f'{env_paths.VALID_VERTS_NARROW}')
        else:
            self.valid_verts = np.load(f'{env_paths.VALID_VERTS}')
        self.can_uv = torch.from_numpy(np.load(env_paths.FLAME_UV_COORDS)[self.valid_verts, :]).cuda().unsqueeze(0).float()
        self.can_uv[..., 1] = (self.can_uv[..., 1] * -1) + 1

        self.verts_2d = []
        self.gt_2_verts = None

        self.valid_vertex_index = torch.from_numpy(self.valid_verts).long().cuda()



    def finish_stage1(self, delta_uv_fine=None, dist_uv_fine=None):
        self.verts_2d = torch.cat(self.verts_2d, dim=0)
        if delta_uv_fine is not None:
            self.delta = delta_uv_fine
            self.dist_uv = dist_uv_fine

    def is_next(self):
        self.gt_2_verts = None

    @torch.compiler.disable
    def compute_corresp(self, gt, selected_frames=None):

        self.gt = gt

        gt_uv = gt[:, :2, :, :].permute(0, 2, 3, 1)
        gt_uv = gt_uv.reshape(gt_uv.shape[0], -1, 2)  # B x n_pixel x 2
        can_uv = self.can_uv.repeat(gt_uv.shape[0], 1, 1)

        knn_result = knn_points(can_uv, gt_uv)
        pixel_position_width = knn_result.idx % gt.shape[-1]
        pixel_position_height = knn_result.idx // gt.shape[-2]
        self.dists = knn_result.dists.clone()

        self.gt_2_verts = torch.cat([pixel_position_width, pixel_position_height], dim=-1)
        if selected_frames is None:
            self.verts_2d.append(torch.cat([pixel_position_width, pixel_position_height], dim=-1))



    def compute_loss(self, proj_vertices, is_visible_verts_idx=None, selected_frames=None, uv_map=None, l2_loss=False):


        if is_visible_verts_idx is not None:
            not_occluded = is_visible_verts_idx[:, self.valid_vertex_index].float()
        else:
            not_occluded = torch.ones_like(self.valid_vertex_index).float().unsqueeze(0)


        if selected_frames is not None:
            gt_2_verts = self.verts_2d[selected_frames, :, :]
        else:
            gt_2_verts = self.gt_2_verts

        valid_proj_v = proj_vertices[:, self.valid_vertex_index, ..., :2]
        v_dist_2d = (gt_2_verts - valid_proj_v)
        if l2_loss:
            uv_loss = (
                ( v_dist_2d/ self.gt.shape[-1]) * (self.dists < self.delta) *
                (v_dist_2d.abs().sum(dim=-1) < self.dist_uv).unsqueeze(-1) *
                not_occluded.unsqueeze(-1)
            ).square().mean() * 100
        else:
            uv_loss = (
                    (v_dist_2d / self.gt.shape[-1]) * (self.dists < self.delta) *
                    (v_dist_2d.abs().sum(dim=-1) < self.dist_uv).unsqueeze(-1) *
                    not_occluded.unsqueeze(-1)
            ).abs().mean()
        return uv_loss


    def compute_loss_lstsq(self, proj_vertices):
        uv_loss = ((self.gt_2_verts / self.gt.shape[-1] - proj_vertices[:, torch.from_numpy(self.valid_verts).long().cuda(), :] /
                    self.gt.shape[-1]) * (self.dists < self.delta)   *
                    (
                               (self.gt_2_verts - proj_vertices[:, torch.from_numpy(self.valid_verts).long().cuda(), :]).abs().sum(
                                   dim=-1) < 30).unsqueeze(-1)
                   )
        #print(uv_loss)
        return uv_loss
