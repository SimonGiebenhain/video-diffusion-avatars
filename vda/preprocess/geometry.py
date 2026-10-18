"""Geometry + rasterisation helpers for the proxy conditioning (the maths the training conditionings were
rendered with): nvdiffrast Lambert renderer, sim3/rigid solvers, FLAME2020 forward via pixel3dmm,
per-identity constant head trim, MHR body with the head removed."""
import numpy as np
import torch

from .. import paths

ASSETS = paths.ASSETS
OUT_W, OUT_H = 720, 1280             # conditioning canvas (the model's native 9:16 crop)
CHIN, BRIDGE = 8, 27                 # FLAME-68 landmark indices
L_HIP, R_HIP, NECK = 9, 10, 69       # MHR-70 keypoint indices
#: FLAME vertex indices of the 68 landmarks
LM = np.load(ASSETS / "flame_lm_inds.npy")

# ---------------- MHR body mesh with the head removed (static vertex mask) ----------------
B_FACES_NP = np.load(ASSETS / "mhr_faces.npy").astype(np.int32)
HEAD_RM = np.load(ASSETS / "mhr_head_verts.npy")
BODY_KEEP = np.ones(18439, bool)
BODY_KEEP[HEAD_RM] = False
BIDX = np.where(BODY_KEEP)[0]
_bremap = -np.ones(18439, np.int64)
_bremap[BIDX] = np.arange(len(BIDX))
_bfmask = BODY_KEEP[B_FACES_NP].all(axis=1)
B_FACES_CUT = _bremap[B_FACES_NP[_bfmask]].astype(np.int32)

# ---------------- nvdiffrast Lambert renderer ----------------
_GL = None


def _ctx():
    global _GL
    if _GL is None:
        import nvdiffrast.torch as dr
        _GL = dr.RasterizeCudaContext()
    return _GL


def render_shaded(verts_w, faces_np, K, c2w, W, H, dev="cuda"):
    """-> (gray uint8 [H,W], mask bool [H,W]) Lambert-shaded, key light 25deg above the camera."""
    import nvdiffrast.torch as dr
    V = torch.from_numpy(np.ascontiguousarray(verts_w, np.float32)).to(dev)
    F = torch.from_numpy(np.ascontiguousarray(faces_np, np.int32)).to(dev).contiguous()
    w2c = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(dev)
    Vc = (w2c[:3, :3] @ V.T).T + w2c[:3, 3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    zn, zf = 0.05, 20.0
    x = Vc[:, 0] * (2 * fx / W)
    y = Vc[:, 1] * (2 * fy / H)
    z = Vc[:, 2]
    cxn = (2 * cx / W) - 1.0
    cyn = (2 * cy / H) - 1.0
    clip = torch.stack([x + cxn * z, -(y + cyn * z),
                        z * (zf + zn) / (zf - zn) - 2 * zf * zn / (zf - zn), z], -1)
    rast, _ = dr.rasterize(_ctx(), clip[None].contiguous(), F, resolution=[H, W])
    v0, v1, v2 = Vc[F[:, 0].long()], Vc[F[:, 1].long()], Vc[F[:, 2].long()]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    vn = torch.zeros_like(Vc)
    for k in range(3):
        vn.index_add_(0, F[:, k].long(), fn)
    vn = vn / (vn.norm(dim=1, keepdim=True) + 1e-9)
    n_pix, _ = dr.interpolate(vn[None].contiguous(), rast.contiguous(), F)
    n_pix = n_pix[0] / (n_pix[0].norm(dim=-1, keepdim=True) + 1e-9)
    n_pix = torch.where(n_pix[..., 2:3] > 0, -n_pix, n_pix)
    e = np.deg2rad(25.0)
    light = torch.tensor([0.0, -np.sin(e), -np.cos(e)], device=dev, dtype=torch.float32)
    lam = (n_pix @ light).clamp(min=0.0)
    shade = (0.32 + 0.72 * lam).clamp(0, 1) * 255.0
    mask = rast[0, ..., 3] > 0
    gray = torch.where(mask, shade, torch.zeros_like(shade))
    # nvdiffrast rasterises in OpenGL row order (row 0 = NDC bottom): flip to image convention
    return (np.ascontiguousarray(gray.cpu().numpy()[::-1]).astype(np.uint8),
            np.ascontiguousarray(mask.cpu().numpy()[::-1]))


def crop_cam(K_native, H_native, W_native):
    """Conditioning crop convention: resize to H=1280 then centre-crop W=720. Returns K'."""
    s = OUT_H / H_native
    K = K_native.copy()
    K[:2, :] *= s
    left = (max(OUT_W, round(W_native * s)) - OUT_W) / 2.0
    K[0, 2] -= left
    return K


def rigid(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    C = (Y - my).T @ (X - mx)
    U, S, Vt = np.linalg.svd(C)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, my - R @ mx


def sim3_align(X, Y):
    """Umeyama similarity X -> Y: (R, t, s) with Y ~= s*(R@X.T).T + t."""
    mx, my = X.mean(0), Y.mean(0)
    C = (Y - my).T @ (X - mx) / len(X)
    U, S, Vt = np.linalg.svd(C)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((X - mx) ** 2).sum() * len(X)
    return R, my - s * R @ mx, s


# ---------------- FLAME2020 (pixel3dmm's implementation: 300 shape / 100 expr, 6D poses, eyelids) ----------------
_flame20 = None


def flame20():
    global _flame20
    if _flame20 is None:
        from omegaconf import OmegaConf
        paths.setup_pixel3dmm_import()              # pixel3dmm's env_paths reads PIXEL3DMM_* at import
        from pixel3dmm.tracking.flame.FLAME import FLAME
        cfg = OmegaConf.create({"use_flame2023": False, "num_shape_params": 300, "num_exp_params": 100})
        _flame20 = FLAME(cfg).cuda()
    return _flame20


def _to6(v):
    from pytorch3d.transforms import matrix_to_rotation_6d, axis_angle_to_matrix
    return matrix_to_rotation_6d(axis_angle_to_matrix(torch.from_numpy(np.asarray(v, np.float32))[None].cuda()))


def neutral20(shape):
    """FLAME2020 neutral canonical mesh of an identity (shape only) -- the head-trim reference."""
    fl = flame20()
    z3 = np.zeros(3, np.float32)
    with torch.no_grad():
        v, _, _, _, _ = fl(
            cameras=torch.eye(3).cuda()[None],
            shape_params=torch.from_numpy(np.asarray(shape, np.float32).reshape(1, -1)).cuda(),
            expression_params=torch.zeros(1, 100).cuda(),
            eye_pose_params=torch.cat([_to6(z3), _to6(z3)], 1),
            jaw_pose_params=_to6(z3), neck_pose_params=None, eyelid_params=None)
    return v[0].cpu().numpy().astype(np.float64)


def fwd20(shape, p, neck=True):
    """FLAME2020 forward for one frame's parameters p = {expression, jaw, eyes, neck[, eyelids]}."""
    fl = flame20()
    with torch.no_grad():
        v, _, _, _, _ = fl(
            cameras=torch.eye(3).cuda()[None],
            shape_params=torch.from_numpy(np.asarray(shape, np.float32).reshape(1, -1)).cuda(),
            expression_params=torch.from_numpy(np.asarray(p["expression"], np.float32)[None]).cuda(),
            eye_pose_params=torch.cat([_to6(p["eyes"][:3]), _to6(p["eyes"][3:])], 1),
            jaw_pose_params=_to6(p["jaw"]),
            neck_pose_params=_to6(p["neck"]) if neck else None,
            eyelid_params=(torch.from_numpy(np.asarray(p["eyelids"], np.float32)[None]).cuda()
                           if p.get("eyelids") is not None else None))
    return v[0].cpu().numpy().astype(np.float64)


def neckless20(shape, p):
    """Head without the neck joint's articulation, rigidly re-anchored on the inner-51 landmarks."""
    V0 = fwd20(shape, p, neck=False)
    Vp = fwd20(shape, p, neck=True)
    R, t = rigid(V0[LM][17:], Vp[LM][17:])
    return (R @ V0.T).T + t


def trim_from_mask(keep, faces):
    idx = np.where(keep)[0]
    remap = -np.ones(len(keep), np.int64)
    remap[idx] = np.arange(len(idx))
    return idx, remap[faces[keep[faces].all(axis=1)]].astype(np.int32)


def head_trim(Vneut, faces):
    """Per-identity CONSTANT head cut (1 cm below the chin along chin->bridge) on the neutral mesh."""
    up = Vneut[LM][BRIDGE] - Vneut[LM][CHIN]
    up = up / np.linalg.norm(up)
    keep = (Vneut - (Vneut[LM][CHIN] - 0.01 * up)) @ up > 0.0
    return trim_from_mask(keep, faces)
