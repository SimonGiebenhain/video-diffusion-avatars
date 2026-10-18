# video-diffusion-avatars code, CC BY-NC-SA 4.0 (LICENSE at the repository root). It builds on the
# DiffSynth-Studio code in this folder (Apache-2.0, vda/third_party/diffsynth/LICENSE).
"""WanAnimateAdapter driven by X-NeMo expression codes (video-diffusion-avatars release).

The stock Wan-Animate adapter encodes a 512x512 face crop with its LIA-style `motion_encoder`
into a 512-d motion code per frame, which `face_encoder` turns into motion tokens that the
`face_adapter` fuser blocks cross-attend to inside the DiT. Here the motion code is instead

    raw X-NeMo code [T, 512]
      --pre  (FROZEN X-NeMo expression disentangler encoder, xnemo_disentangler.pt)--> z_expr
      --f    (trained 512->1024->1024->1024->512 GELU MLP, in the adapter checkpoint)--> WAN code
      --face_encoder / face_adapter (fine-tuned, in the adapter checkpoint)--> fuser tokens

so the viewpoint-dependent part of the X-NeMo code never reaches the video model. A second,
zero-initialised patch embedding (`graft_patch_embedding`) adds the FLAME-head + body "graft"
render (= pose_proxy.mp4, the "proxy render" of the docs) to the skeleton pose tokens.

Only the released configuration is implemented: one translator MLP `f`, the frozen disentangler encoder inside
the graph, and Wan-Animate's native (ungated) fuser residual.
"""

import torch
import torch.nn as nn

from .wan_video_animate_adapter import WanAnimateAdapter


class TranslatorHead(nn.Module):
    """raw X-NeMo code -> [frozen pre] -> z_expr -> f -> native WAN motion code (fp32)."""

    def __init__(self, f, pre, pre_unit_norm=False):
        super().__init__()
        self.pre = pre
        self.pre_unit_norm = bool(pre_unit_norm)
        for p in self.pre.parameters():
            p.requires_grad_(False)
        self.f = f

    def forward(self, x):                       # [B, T, 512] raw X-NeMo codes
        B, T, D = x.shape
        z = x.reshape(-1, D).float()
        # The CFG null is an exactly-zero code (training cfg-dropout and the inference negative /
        # base passes both zero expr_codes). pre(0) is the encoder's bias response, NOT a null,
        # so a zero code is passed through as zero: null in, null out.
        _null = (z.abs().sum(dim=-1, keepdim=True) == 0)
        z = self.pre(z)
        if self.pre_unit_norm:
            z = torch.nn.functional.normalize(z, dim=-1)
        z = z.masked_fill(_null, 0.0)
        return self.f(z).reshape(B, T, -1).to(x.dtype)


def load_xnemo_pre(xnemo_disent_ckpt, verbose=True):
    """FROZEN X-NeMo disentangler expression encoder (+ its expr_unit_norm flag) from the
    disentangler checkpoint: MLP shape from the ckpt cfg (embedding_dim -> expr_dim,
    n_layers/width, ReLU), weights from the 'expr_encoder.*' subtree of model_state."""
    ck = torch.load(xnemo_disent_ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])

    def _get(node, key, default=None):
        return node.get(key, default) if hasattr(node, "get") else getattr(node, key, default)

    D = int(_get(cfg, "embedding_dim", None) or 512)
    expr_dim = int(_get(cfg, "expr_dim"))
    enc_cfg = _get(cfg, "expr_encoder")
    n_layers, width = int(_get(enc_cfg, "n_layers")), int(_get(enc_cfg, "width"))
    L, c = [], D
    for _ in range(n_layers - 1):
        L += [nn.Linear(c, width), nn.ReLU()]
        c = width
    L.append(nn.Linear(c, expr_dim))
    pre = nn.Sequential(*L)
    sub = {k[len("expr_encoder."):]: v for k, v in ck["model_state"].items() if k.startswith("expr_encoder.")}
    pre.load_state_dict(sub, strict=True)
    unit = bool(_get(cfg, "expr_unit_norm", False))
    if verbose:
        print(f"[translator] frozen X-NeMo disentangler pre-encoder ({D}->{expr_dim}, {n_layers} layers, "
              f"unit_norm={unit}) from {xnemo_disent_ckpt}", flush=True)
    return pre, unit


#: architecture of the translator MLP `f` (its weights come from the adapter checkpoint)
TRANSLATOR_F = dict(in_dim=512, out_dim=512, width=1024, layers=4)


def build_translator_head(xnemo_disent_ckpt, device):
    def _mlp(i, o, w, n):
        L, d = [], i
        for _ in range(n - 1):
            L += [nn.Linear(d, w), nn.GELU()]
            d = w
        return nn.Sequential(*(L + [nn.Linear(d, o)]))

    f = _mlp(TRANSLATOR_F["in_dim"], TRANSLATOR_F["out_dim"], TRANSLATOR_F["width"], TRANSLATOR_F["layers"])
    pre, pre_unit = load_xnemo_pre(xnemo_disent_ckpt)
    return TranslatorHead(f, pre, pre_unit_norm=pre_unit).to(device).float()


class WanAnimateAdapterXNemo(WanAnimateAdapter):
    """Obtained by class reassignment on the loaded stock adapter (keeps its pretrained
    motion_encoder / pose_patch_embedding), then `translator` and `graft_patch_embedding` are
    attached and the fine-tuned checkpoint is loaded on top (see vda.model.load_pipeline)."""

    def after_patch_embedding_xnemo(self, x, pose_latents, expr_codes, graft_latents=None, frame_offset=0):
        # KV block forward: embed only the pose/graft rows this block reads. Both patch embeddings are
        # Conv3d with a temporal kernel of 1, so slicing first is exact (and avoids embedding the whole
        # video every block). Full-sequence calls take the plain path.
        _row0 = 0
        _lo, _hi = max(int(frame_offset), 1) - 1, int(frame_offset) + int(x.shape[2]) - 1
        if pose_latents is not None and (_lo > 0 or _hi < pose_latents.shape[2]):
            _hi = max(_hi, _lo + 1)
            _row0 = _lo
            pose_latents = pose_latents[:, :, _lo:_hi]
            if graft_latents is not None:
                graft_latents = graft_latents[:, :, _lo:_hi]
        pose_latents = self.pose_patch_embedding(pose_latents)
        # graft branch: FLAME head + body render (Lambert gray blended with the per-pixel ray direction), VAE-encoded
        # like the skeleton; graft_latents=None is the trained graft-absent state (graft guidance).
        if getattr(self, "graft_patch_embedding", None) is not None and graft_latents is not None:
            if graft_latents.shape[2] != pose_latents.shape[2]:
                raise ValueError(f"graft/pose latent length mismatch: {graft_latents.shape[2]} vs "
                                 f"{pose_latents.shape[2]} -- the two conditioning videos must be frame-aligned")
            pose_latents = pose_latents + self.graft_patch_embedding(graft_latents)
        # pose row f-1 belongs to latent frame f >= 1 (slot 0 is the out-of-timeline reference)
        _Tb = x.shape[2]
        _f0 = max(int(frame_offset), 1)
        if _f0 < frame_offset + _Tb:
            x[:, :, _f0 - frame_offset:] += pose_latents[:, :, _f0 - 1 - _row0:frame_offset + _Tb - 1 - _row0]

        expr_codes = self.translator(expr_codes)
        motion_vec = self.face_encoder(expr_codes)  # [B, T_raw, 512] -> [B, T_latent, 5, hidden_dim]
        B, L, H, C = motion_vec.shape
        pad_face = torch.zeros(B, 1, H, C).type_as(motion_vec)
        motion_vec = torch.cat([pad_face, motion_vec], dim=1)  # zero pad for the reference slot
        # the fuser takes the frame count from motion_vec -> hand it this block's frames
        return x, motion_vec[:, frame_offset:frame_offset + _Tb]

    def after_transformer_block(self, block_idx, x, motion_vec):
        if block_idx % 5 == 0:
            residual_out = self.face_adapter.fuser_blocks[block_idx // 5](x, motion_vec)
            x = residual_out + x      # native (ungated) Wan-Animate residual
        return x
