# -*- coding: utf-8 -*-
import os.path

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from skimage.io import imread

import nvdiffrast.torch as dr
#import nvdiffrast_util as util
import pyvista as pv

from torchvision.transforms.functional import gaussian_blur

glctx = dr.RasterizeCudaContext()
#glctx = dr.RasterizeGLContext()


from pixel3dmm.tracking import util
from pixel3dmm.utils import obj_util
from pixel3dmm.utils import utils_3d
from pixel3dmm.utils.masking import Masking

from pixel3dmm import env_paths

sky = torch.from_numpy(np.array([80, 140, 200]) / 255.).cuda()


def apply_gamma(rgb, gamma="srgb"):
    if gamma == "srgb":
        T = 0.0031308
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(rgb < T, 12.92 * rgb, (1.055 * torch.pow(torch.abs(rgb1), 1 / 2.4) - 0.055))
    elif gamma is None:
        return rgb
    else:
        return torch.pow(torch.max(rgb, rgb.new_tensor(0.0)), 1.0 / gamma)


def remove_gamma(rgb, gamma="srgb"):
    if gamma == "srgb":
        T = 0.04045
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(rgb < T, rgb / 12.92, torch.pow(torch.abs(rgb1 + 0.055) / 1.055, 2.4))
    elif gamma is None:
        return rgb
    else:
        res = torch.pow(torch.max(rgb, rgb.new_tensor(0.0)), gamma) + torch.min(rgb, rgb.new_tensor(0.0))
        return res


def transform_pos(mtx, pos):
    #t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], pos.shape[1], 1]).cuda()], axis=2)
    return torch.matmul(posw, mtx.permute(0, 2, 1)) #[None, ...]

class NVDRenderer(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=512, flip=False,
                 no_sh : bool = False,
                 white_bg : bool = False,
                 ):
        super(NVDRenderer, self).__init__()
        #TODO path management
        verts, uv_coords, colors, faces, uv_faces = obj_util.load_obj(f'{env_paths.head_template}')

        self.pos_idx = torch.from_numpy(np.array(faces)).cuda().int()
        self.uv_idx = torch.from_numpy(np.array(uv_faces)).cuda().int()

        self.uv = torch.from_numpy(np.array(uv_coords)).float().cuda()
        self.uv[:, 1] = (self.uv[:, 1] * -1) + 1
        self.uv[:, 0] = (self.uv[:, 0] * -1) + 1

        self.max_mipmap_level = 6
        self.white_bg = white_bg





        self.image_size = image_size
        self.uv_size = uv_size

        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        faces = faces.verts_idx[None, ...]

        self.fg_color = torch.ones([verts.shape[0], 1]).float().cuda()

        mask = torch.from_numpy(imread(f'{env_paths.EYE_MASK}') / 255.).permute(2, 0, 1).cuda()[0:3, :, :]
        mask = mask > 0.
        mask = F.interpolate(mask[None].float(), [2048, 2048], mode='bilinear')

        self.register_buffer('mask', mask)

        self.masking = Masking()
        self.render_mask = self.masking.get_mask_rendering().cuda()
        self.face_mask = self.masking.to_render_mask(self.masking.get_mask_face()).cuda()
        self.eye_mask = self.masking.get_mask_eyes_rendering().cuda()


        # faces
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coordsw
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        #uvcoords[..., 0] = -uvcoords[..., 0]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        # shape colors
        colors = torch.tensor([74, 120, 168])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.
        face_colors = util.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)

        ## lighting
        pi = np.pi
        sh_const = torch.tensor(
            [
                1 / np.sqrt(4 * pi),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
            ],
            dtype=torch.float32,
        )
        self.register_buffer('constant_factor', sh_const)

        self.no_sh = no_sh
        self.rast_out = None
        self.rast_out_db = None


    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = torch.stack([
            N[:, 0] * 0. + 1., N[:, 0], N[:, 1],
            N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
            N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * (N[:, 2] ** 2) - 1
        ], 1)  # [bz, 9, h, w]
        sh = sh * self.constant_factor[None, :, None, None]
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
        return shading

    def reset(self):
        self.rast_out = None
        self.rast_out_db = None

    def forward(self, vertices_world, albedos, lights, r_mvps, R, T,
                aux_texture = None,
                texture_observation_mask = None,
                verts_can=None,
                verts_noneck=None,
                verts_can_can=None,
                verts_depth=None,
                is_viz=False,
                ):
        B = vertices_world.shape[0]
        faces = self.faces.expand(B, -1, -1)

        meshes_world_noneck = Meshes(verts=verts_noneck.float(), faces=faces.long())
        normals = meshes_world_noneck.verts_normals_packed().reshape(B, 5023, 3)


        face_mask = self.face_mask.repeat(B, 1, 1)
        render_mask = self.render_mask.repeat(B, 1, 1) # mask used to define where loss is computed --> should only optimize for texture offsets inside this mask!!!
        eyes_mask = self.eye_mask.repeat(B, 1, 1)

        pos_clips = transform_pos(r_mvps, vertices_world).float()

        if self.rast_out is None:
            rast_out, rast_out_db = dr.rasterize(glctx, pos_clips, self.pos_idx,
                                                 resolution=[self.image_size, self.image_size])
        else:
            rast_out = self.rast_out
            rast_out_db = self.rast_out_db


        texc, texd = dr.interpolate(self.uv, rast_out, self.uv_idx, rast_db=rast_out_db, diff_attrs='all')
        rendered_normals = dr.interpolate(normals, rast_out, self.pos_idx)[0].permute(0, 3, 1, 2)

        rendered_face_mask = dr.interpolate(face_mask, rast_out, self.pos_idx)[0].permute(0, 3, 1, 2)
        rendered_mask = dr.interpolate(render_mask, rast_out, self.pos_idx)[0].permute(0, 3, 1, 2)
        if verts_depth is not None:
            actual_rendered_depth = dr.interpolate(verts_depth.repeat(1, 1, 3), rast_out, self.pos_idx)[0].permute(0, 3, 1, 2)[:, :1, :, :]
        rendered_eyes_mask = dr.interpolate(eyes_mask, rast_out, self.pos_idx)[0].permute(0, 3, 1, 2)




        mask = self.mask.repeat(B, 1, 1, 1)
        mask_images = dr.texture(mask.permute(0, 2, 3, 1).contiguous(), texc, filter_mode='linear') #.permute(0, 3, 1, 2)
        mask_images = dr.antialias(mask_images, rast_out, pos_clips, self.pos_idx).permute(0, 3, 1, 2)

        alpha_images = torch.ones_like(mask_images)

        uv_images = torch.cat([1-texc[..., :1], texc[..., 1:]], dim=-1)
        outputs = {

            'alpha_images': alpha_images,
            'mask_images_mesh': (rendered_face_mask > 0).float(),

            'normal_images': rendered_normals,
            'mask_images': (mask_images > 0).float(),
            'mask_images_rendering': (rendered_mask > 0).float(),
            'mask_images_eyes': (rendered_eyes_mask > 0).float(),
            'uv_images': uv_images,
            'fg_images': mask_images,
        }


        if verts_depth is not None:
            outputs['actual_rendered_depth'] = actual_rendered_depth

        if is_viz:
            rendered_normals_detached = rendered_normals.detach()
            position_images_world_space = dr.interpolate(vertices_world, rast_out, self.pos_idx)[0].permute(0, 3, 1, 2)
            cam_positions = -torch.einsum('bxy,by->bx', R, T)
            viewing_angle = (position_images_world_space - cam_positions.unsqueeze(-1).unsqueeze(-1))
            viewing_angle_image = (
                    -viewing_angle / viewing_angle.norm(dim=1).unsqueeze(1) * rendered_normals_detached).sum(dim=1)
            outputs['alpha_images'] = viewing_angle_image[:, None, :, :].repeat(1, 3, 1, 1)



        return outputs
