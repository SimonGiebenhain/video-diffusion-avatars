# *************************************************************************
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by ByteDance Ltd. and/or its affiliates.
#
# Original file was released under PD-FGC, with the full license text
# available at https://github.com/Dorniwang/PD-FGC-inference/blob/main/LICENSE.
#
# This modified file is released under the same license.
# *************************************************************************
# Modified for video-diffusion-avatars: get_1d_sincos_pos_embed_from_grid reimplemented in numpy (newer diffusers raise on numpy input).
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .FAN_feature_extractor import FAN_SA
from einops import rearrange


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    # numpy implementation of diffusers' (pre-0.34) helper; newer diffusers raise on numpy input.
    # (video-diffusion-avatars: the value only initialises the `pe` buffer, which the checkpoint overwrites.)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", np.asarray(pos).reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
from diffusers.models.modeling_utils import ModelMixin


class MotEncoder_withExtra(ModelMixin):
    def __init__(self, extra_feat_dim=3, out_ch=16):
        super(MotEncoder_withExtra, self).__init__()
        self.model = FAN_SA()
        self.out_drop = None #nn.Dropout(p=0.4)
        self.out_ch = out_ch
        expr_dim = 512
        extra_pos_embed = get_1d_sincos_pos_embed_from_grid(out_ch, np.arange(expr_dim//out_ch))
        self.register_buffer("pe", torch.from_numpy(extra_pos_embed).float().unsqueeze(0))
        self.bbox_proj = nn.Sequential(
            nn.Linear(extra_feat_dim, 64),
            nn.ReLU(),
            nn.Identity(),
            nn.Linear(64, 64),
        )
        self.final_proj = nn.Linear(expr_dim + 64, expr_dim)
        self.out_bn = None

    def change_out_dim(self, out_ch):
        self.out_proj = nn.Linear(self.out_ch, out_ch)

    def forward(self, x, emb):
        if x.ndim == 5:
            latent = self.model(rearrange(x, "b c f h w -> (b f) c h w"))
            emb = rearrange(emb, "b f c -> (b f) c")
            vid_len = x.shape[2]
        else:
            vid_len = 1
            latent = self.model(x)  # [B, 512]
        if self.out_bn is not None:
            latent = self.out_bn(latent.unsqueeze(-1)).squeeze(-1)        #####
        if self.out_drop is not None:
            latent = self.out_drop(latent)
        face_mot_feat = latent.clone()
        latent = torch.cat([latent, self.bbox_proj(emb)], dim=1)
        latent = self.final_proj(latent)
        latent = rearrange(latent, "b (l c) -> b l c", c=self.out_ch) + self.pe
        
        if x.ndim == 5:
            latent = rearrange(latent, "(b f) l c -> b f l c", f=x.shape[2])
            face_mot_feat = rearrange(face_mot_feat, "(b f) c -> b f c", f=x.shape[2])

        return latent

    def encode_facemot(self, x):
        if x.ndim == 5:
            latent = self.model(rearrange(x, "b c f h w -> (b f) c h w"))
        else:
            latent = self.model(x)  # [B, 512]
        if self.out_bn is not None:
            latent = self.out_bn(latent.unsqueeze(-1)).squeeze(-1)  #####
        if x.ndim == 5:
            latent = rearrange(latent, "(b f) c -> b f c", f=x.shape[2])
        return latent

    def decode_facemot(self, mot_tok, emb):
        pred_motion = self.vqvae.forward_decoder(mot_tok)
        b, t = pred_motion.shape[:2]
        c = 512
        split_exp_num = c // self.vqvae.vqvae.input_emb_width
        latent = pred_motion.reshape(b, t, split_exp_num, c // split_exp_num).reshape(b * t, c)

        emb = rearrange(emb.repeat(1, t, 1), "b f c -> (b f) c")
        latent = torch.cat([latent, self.bbox_proj(emb)], dim=1)
        latent = self.final_proj(latent)
        latent = rearrange(latent, "b (l c) -> b l c", c=self.out_ch) + self.pe
        if self.out_proj is not None:
            latent = self.out_proj(latent)

        latent = rearrange(latent, "(b f) l c -> b f l c", f=t)
        return latent

    def fuse_emb(self, x, emb):
        if x.ndim == 3:
            latent = rearrange(x, "b f c -> (b f) c")
            emb = rearrange(emb, "b f c -> (b f) c")
        else:
            latent = x
        latent = torch.cat([latent, self.bbox_proj(emb)], dim=1)
        latent = self.final_proj(latent)
        latent = rearrange(latent, "b (l c) -> b l c", c=self.out_ch) + self.pe
        if self.out_proj is not None:
            latent = self.out_proj(latent)

        if x.ndim == 3:
            latent = rearrange(latent, "(b f) l c -> b f l c", f=x.shape[1])

        return latent
