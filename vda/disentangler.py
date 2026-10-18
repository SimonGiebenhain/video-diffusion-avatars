"""X-NeMo expression disentangler (standalone).

X-NeMo motion codes [512] are view-DEPENDENT: the same expression seen from two cameras gives
different codes (codes of one frame from two cameras are less similar than codes of consecutive frames from one
camera). The disentangler splits a code into
    z_expr [512]  view-invariant expression   (expr_encoder)
    z_cam  [32]   viewpoint / camera           (cam_encoder)
and entangles them back into the X-NeMo code space (decoder, [z_expr | z_cam] -> 512). Trained
contrastively on NeRSemble (same frame across 16 cameras = same expression class, same camera =
same view class) plus reconstruction and cycle losses.

Inside the video model only `expr_encoder` is used (frozen, as the first stage of the translator:
raw X-NeMo -> z_expr -> WAN motion code). Standalone uses:

    from vda.disentangler import XNemoDisentangler
    dis = XNemoDisentangler.load("weights/xnemo_disentangler.pt")
    z_expr, z_cam = dis.disentangle(codes)                 # codes: [T, 512] raw X-NeMo
    codes_novel = dis.entangle(z_expr, z_cam_target)       # re-render the expression from another view

    python scripts/disentangle.py codes.npy --out z.npz       # z_expr / z_cam / reconstruction for a code file
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(in_dim, out_dim, n_layers, width):
    layers, cur = [], in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(cur, width), nn.ReLU()]
        cur = width
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


class XNemoDisentangler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = int(cfg.get("embedding_dim") or 512)
        self.expr_unit_norm = bool(cfg.get("expr_unit_norm", False))
        self.expr_encoder = build_mlp(D, cfg["expr_dim"], cfg["expr_encoder"]["n_layers"], cfg["expr_encoder"]["width"])
        self.cam_encoder = build_mlp(D, cfg["cam_dim"], cfg["cam_encoder"]["n_layers"], cfg["cam_encoder"]["width"])
        if cfg.get("id_dim", 0) or cfg.get("res_dim", 0):
            raise ValueError("this release supports the expr+cam disentangler (id_dim = res_dim = 0)")
        self.decoder = build_mlp(cfg["expr_dim"] + cfg["cam_dim"], D, cfg["decoder"]["n_layers"], cfg["decoder"]["width"])
        self.cfg = cfg

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(dict(ck["cfg"]))
        m.load_state_dict(ck["model_state"], strict=True)
        return m.to(device).eval()

    @torch.no_grad()
    def expr_code(self, z):
        z_expr = self.expr_encoder(z)
        return F.normalize(z_expr, dim=-1) if self.expr_unit_norm else z_expr

    @torch.no_grad()
    def disentangle(self, codes):
        z = self._t(codes)
        out = self.expr_code(z), self.cam_encoder(z)
        return tuple(self._back(o, codes) for o in out)

    @torch.no_grad()
    def entangle(self, z_expr, z_cam):
        ze, zc = self._t(z_expr), self._t(z_cam)
        if zc.shape[0] == 1 and ze.shape[0] > 1:
            zc = zc.expand(ze.shape[0], -1)
        return self._back(self.decoder(torch.cat([ze, zc], dim=-1)), z_expr)

    def _t(self, x):
        dev = next(self.parameters()).device
        x = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x, dtype=torch.float32, device=dev)
        return x[None] if x.dim() == 1 else x

    @staticmethod
    def _back(y, like):
        return y.cpu().numpy() if not torch.is_tensor(like) else y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("codes", help="[T, 512] RAW X-NeMo codes .npy")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True, help=".npz with z_expr [T,512], z_cam [T,32], recon [T,512]")
    a = ap.parse_args()
    from . import paths
    dis = XNemoDisentangler.load(a.ckpt or paths.weight(paths.DISENTANGLER))
    codes = np.load(a.codes).astype(np.float32)
    z_expr, z_cam = dis.disentangle(codes)
    recon = dis.entangle(z_expr, z_cam)
    cos = (recon * codes).sum(1) / (np.linalg.norm(recon, axis=1) * np.linalg.norm(codes, axis=1) + 1e-8)
    np.savez(a.out, z_expr=z_expr, z_cam=z_cam, recon=recon)
    print(f"[vda] {len(codes)} codes: |raw| {np.linalg.norm(codes, axis=1).mean():.1f}, |z_expr| "
          f"{np.linalg.norm(z_expr, axis=1).mean():.2f}, reconstruction cosine {cos.mean():.3f} -> {a.out}")


if __name__ == "__main__":
    main()
