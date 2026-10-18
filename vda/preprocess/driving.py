"""Driving sequence from an arbitrary monocular video ("in-the-wild driving").

The novel-view conditioning lives in the canonical person's world (see vda/preprocess/camera.py),
so a driver video does not need to be placed in 3D. From the video we only take what the model
should copy -- the FACIAL performance:
  * FLAME2020 expression [100], jaw, eyes, eyelids per frame (single-camera pixel3dmm tracking,
    camera estimated by the tracker),
  * RAW X-NeMo codes [512] per frame (vda.preprocess.xnemo_codes),
and put it on a canonical take of the driving bank, which supplies head placement, neck, body and
skeleton (`--motion take` = that take's motion, looped back and forth to the video length;
`--motion static` = one frame of it, frozen). The reference identity is swapped in later
(vda.preprocess.conditioning --ref), exactly as for the bank's own sequences.

    python prepare_driving.py --video drv.mp4 --name drv --take 1 --motion take
    -> cache/driving/drv/{frames/, codes_raw.npy, mono_flame.npz, state.npz}
    then: python render_conditioning.py --ref me --driving drv ...

`--motion driver` transfers the DRIVER's head + upper-body motion as well, the way the bank's own takes
were made: SAM 3D Body's per-frame MHR pose of the video (vda.preprocess.body_motion, separate env;
temporally smoothed here, --motion_sigma) re-poses the canonical person's MHR body -- 130-d body pose
(spine / neck / head) + root rotation relative to the clip's mean root, composed onto the canonical
person's base root, pelvis pinned -- and the FLAME head rides rigidly on that body's head frame (MHR
keypoints nose / ears), so head, neck and torso always agree. Needs <canonical>_canonical_body.npz (shipped next
to the bank: the canonical person's MHR identity, scales, pinned pelvis, model->world similarity, base
keypoints + head landmarks) and SAM3D's MHR model file (mhr_model.pt, path stored by body_motion.py).

    python prepare_driving.py --video drv.mp4 --name drv  # frames, codes, FLAME
    python -m vda.preprocess.body_motion --name drv --sam3d /path/to/sam-3d-body              # SAM 3D Body env
    python prepare_driving.py --video drv.mp4 --name drv --motion driver

With `take` / `static` the driver's own head rotation is NOT transferred (the head would move against a
canonical body); expression, jaw, gaze and blinks are.
Needs the `vda` env with pixel3dmm and diffusers (X-NeMo encoder), a GPU.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .. import paths
from .reference import TRACK_SUFFIX, pixel3dmm_runner

FACE_KEYS = ("expression", "jaw", "eyes", "eyelids")      # taken from the driver video


def extract_frames(video, out_dir, fps=None):
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    vf = ["-vf", f"fps={fps}"] if fps else []
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), *vf, "-q:v", "2", str(out_dir / "%05d.jpg")], check=True)
    return sorted(out_dir.glob("*.jpg"))


def track_mono(frames_dir, name, work, min_conf=None, batch_size=8):
    """pixel3dmm single-camera video tracking -> per-frame FLAME2020 params (axis-angle)."""
    import torch
    from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
    work = Path(work)
    if work.exists():
        shutil.rmtree(work)                      # stages skip themselves on leftovers: always start clean
    src = work / "src" / name
    shutil.copytree(frames_dir, src)
    run = pixel3dmm_runner(work)
    run("scripts/run_cropping.py", "--video_or_images_path", str(src),
        *(["--min_conf", str(min_conf)] if min_conf else []))
    run("scripts/run_facer_segmentation.py", "--video_name", name)
    for t in ("normals", "uv_map"):
        run("scripts/network_inference.py", f"model.prediction_type={t}", f"video_name={name}")
    run("scripts/track.py", f"video_name={name}", f"batch_size={batch_size}", "cache_on_cpu=true")
    ck = sorted((work / "tracking" / f"{name}{TRACK_SUFFIX}" / "checkpoint").glob("*.frame"))
    if not ck:
        raise SystemExit(f"tracker wrote no checkpoints under {work / 'tracking'}")
    P = {k: [] for k in ("frames", "expression", "jaw", "eyes", "neck", "eyelids")}
    aa = lambda r6: matrix_to_axis_angle(rotation_6d_to_matrix(torch.as_tensor(np.asarray(r6, np.float32)))).numpy()  # noqa: E731
    for f in ck:
        fl = torch.load(f, weights_only=False, map_location="cpu")["flame"]
        P["frames"].append(int(f.stem))
        P["expression"].append(np.asarray(fl["exp"])[0])
        P["jaw"].append(aa(np.asarray(fl["jaw"]).reshape(1, 6))[0])
        P["neck"].append(aa(np.asarray(fl["neck"]).reshape(1, 6))[0])
        e = np.asarray(fl["eyes"]).reshape(2, 6)
        P["eyes"].append(aa(e).reshape(6))
        P["eyelids"].append(np.asarray(fl["eyelids"])[0][:2])
    return {k: np.asarray(v, np.float32 if k != "frames" else np.int64) for k, v in P.items()}


def canonical_motion(canon, take, T, motion):
    """Frame indices into the canonical state for T output frames."""
    sel = np.arange(len(canon["codes"]))
    if take:
        sel = np.where(canon["timeline"] == f"take{take}")[0]
        if not len(sel):
            raise SystemExit(f"no take{take} in the canonical sequence")
    if motion == "static":
        return np.full(T, sel[0])
    n = len(sel)
    period = np.r_[np.arange(n), np.arange(n - 2, 0, -1)] if n > 1 else np.zeros(1, int)   # ping-pong
    return sel[period[np.arange(T) % len(period)]]


def build_state(canon, idx, mono, codes):
    T = len(idx)
    if len(codes) != T or len(mono["expression"]) != T:
        raise SystemExit(f"length mismatch: video {T}, codes {len(codes)}, tracked {len(mono['expression'])}")
    st = {}
    for k, v in canon.items():
        st[k] = v[idx] if (np.ndim(v) >= 1 and len(v) == len(canon["codes"]) and k != "rig") else v
    for k in FACE_KEYS:
        if f"z_{k}" in st:
            st[f"z_{k}"] = mono[k].astype(st[f"z_{k}"].dtype)
    st["codes"] = np.asarray(codes, np.float32)
    st["timeline"] = np.array(["video"] * T)
    return st


# ---------------- --motion driver: the driver's MHR body pose re-posed on the canonical person ----------------
FLIP = np.array([1.0, -1.0, -1.0])               # MHR model frame -> SAM3D camera frame (y, z negated)
NOSE, L_EAR, R_EAR = 0, 3, 4                     # MHR-70 keypoints spanning the head frame


def _rotm(e):
    """MHR root rotation (SAM3D's `global_rot`: Euler angles, EXTRINSIC x-y-z = scipy "xyz"; checked by a rigid fit
    of the posed MHR mesh, 3e-6 deg) -> matrix."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", np.asarray(e, np.float64)).as_matrix()


def _euler(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, np.float64)).as_euler("xyz")


def mean_rot(Rs):
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    return U @ Vt


def smooth_rotations(R, sigma):
    """Gaussian in time on the matrix entries, re-projected onto SO(3)."""
    from scipy.ndimage import gaussian_filter1d
    M = gaussian_filter1d(np.asarray(R, np.float64), sigma, axis=0, mode="nearest")
    U, _, Vt = np.linalg.svd(M)
    D = np.ones((len(M), 3))
    D[:, 2] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[:, None, :]) @ Vt


def head_frame(kp):
    """Head frame of MHR-70 keypoints: origin between the ears, x = right->left ear, z = towards the nose."""
    o = 0.5 * (kp[L_EAR] + kp[R_EAR])
    x = kp[L_EAR] - kp[R_EAR]
    x = x / np.linalg.norm(x)
    f = kp[NOSE] - o
    z = f - (f @ x) * x
    z = z / np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], 1), o


def rot_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def fill_missing(x, ok):
    """Frames without a SAM3D detection take the nearest detected frame's value."""
    ok = np.asarray(ok, bool)
    if ok.all():
        return x
    good = np.where(ok)[0]
    near = good[np.abs(np.arange(len(ok))[:, None] - good[None]).argmin(1)]
    return x[near]


def canonical_forward(canon, root, body_pose, mhr_path, km70, device=None, chunk=64):
    """Pose the canonical person's MHR body (identity c_fit, base-frame scales, pinned pelvis) with per-frame
    root rotations [T,3] (MHR Euler xyz) + body poses [T,130] and place it in the canonical world (same maths as
    used to build the shipped driving bank). -> verts [T,18439,3], kp70 [T,70,3] (metres)."""
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mhr = torch.jit.load(str(mhr_path), map_location=dev).eval()
    tt = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=dev)          # noqa: E731
    T = len(root)
    mp = torch.cat([tt(canon["pelvis"])[None].expand(T, -1) * 10, tt(root), tt(body_pose)[:, :130],
                    tt(canon["scales_b"])[None].expand(T, -1)], 1)
    shape = tt(canon["c_fit"])[None].expand(T, -1).contiguous()
    flip, s, R, t = tt(FLIP), tt(canon["sim_s"]), tt(canon["sim_R"]), tt(canon["sim_t"])
    km = tt(km70)
    V, K = [], []
    with torch.no_grad():
        for i in range(0, T, chunk):
            m = mp[i:i + chunk]
            v, jm = mhr(shape[i:i + chunk], m, torch.zeros(len(m), 72, device=dev))
            w = s * torch.einsum("ij,tnj->tni", R, v / 100.0 * flip) + t
            jw = s * torch.einsum("ij,tnj->tni", R, jm[..., :3] / 100.0 * flip) + t
            V.append(w.cpu().numpy())
            K.append(torch.einsum("av,tvj->taj", km, torch.cat([w, jw], 1)).cpu().numpy())
    return np.concatenate(V).astype(np.float64), np.concatenate(K).astype(np.float64)


def driver_root_and_pose(body, sel, sigma, root_b):
    """SAM3D per-frame pose -> (canon root [T,3] MHR Euler xyz, body pose [T,130], detected mask): missed frames
    filled, smoothed in time, root motion taken RELATIVE to the clip's mean root and composed onto the
    canonical person's base root."""
    from scipy.ndimage import gaussian_filter1d
    ok = np.asarray(body["ok"], bool)[sel]
    if not ok.any():
        raise SystemExit("SAM3D detected no person in the tracked frames")
    Rc = _rotm(fill_missing(np.asarray(body["global_rot"], np.float64)[sel], ok))
    bp = np.unwrap(fill_missing(np.asarray(body["body_pose"], np.float64)[sel][:, :130], ok), axis=0)
    if sigma > 0:
        Rc = smooth_rotations(Rc, sigma)
        bp = gaussian_filter1d(bp, sigma, axis=0, mode="nearest")
    root = _euler(_rotm(root_b) @ mean_rot(Rc).T @ Rc)
    return root, bp, ok


def head_placement(canon, kp, mono):
    """68 head-placement landmarks per frame: the canonical person's base-frame head placement, moved rigidly
    with the transferred body's head frame. The placement mesh carries the driver's own expression / jaw (a
    FLAME2020 head of the canonical person's shape, sim3-registered once onto the base landmarks at neutral
    expression), so the renderer's landmark sim3 sees no expression-dependent pull."""
    from . import geometry as G
    shape = np.asarray(canon["flame20_shape"], np.float32)
    keys = [k for k in ("expression", "jaw", "eyes", "eyelids") if k in mono]
    z0 = {k: np.zeros_like(mono[k][0]) for k in keys}
    z0["neck"] = np.zeros(3, np.float32)
    Rs, ts, ss = G.sim3_align(G.neckless20(shape, z0)[G.LM], np.asarray(canon["gtw_lm_b_neutral"], np.float64))
    RHb, ob = head_frame(np.asarray(canon["kp_b"], np.float64))
    L, ang = [], []
    for i in range(len(kp)):
        p = {k: mono[k][i] for k in keys}
        p["neck"] = np.zeros(3, np.float32)
        P = ss * (Rs @ G.neckless20(shape, p)[G.LM].T).T + ts
        RH, o = head_frame(kp[i])
        RM = RH @ RHb.T
        L.append((RM @ P.T).T + o - RM @ ob)
        ang.append(rot_deg(RM))
    return np.stack(L), np.asarray(ang)


def driver_motion(body, canon, sel, sigma, mhr_path, mono):
    """-> verts [T,Nb,3] (body without head), kp [T,70,3], gtw_lm [T,68,3] in the canonical world, stats."""
    from . import geometry as G
    root, bp, ok = driver_root_and_pose(body, sel, sigma, canon["root_b"])
    W, kp = canonical_forward(canon, root, bp, mhr_path, body["kp_map70"])
    sc = float(canon["ik_scale"])
    W, kp = W * sc, kp * sc
    gtw, ang = head_placement(canon, kp, mono)
    RH = np.stack([head_frame(k)[0] for k in kp])
    step = np.r_[0.0, rot_deg(np.einsum("tji,tjk->tik", RH[:-1], RH[1:]))]
    stats = {"detected": float(ok.mean()), "head_rot_vs_base_deg": [float(ang.min()), float(ang.max())],
             "head_step_deg_median": float(np.median(step)), "head_step_deg_p99": float(np.percentile(step, 99)),
             "head_step_deg_max": float(step.max())}
    return W[:, G.BIDX].astype(np.float32), kp, gtw, stats


def default_canonical_body(canonical):
    p = Path(canonical)
    return p.with_name(p.stem + "_canonical_body.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--canonical", default="nersemble8",
                    help="driving sequence whose canonical person carries the performance (head / body motion; default "
                         "the shipped nersemble8)")
    ap.add_argument("--take", type=int, default=1)
    ap.add_argument("--motion", choices=["take", "static", "driver"], default="take",
                    help="head/body motion: canonical take (looped), one canonical frame, or the DRIVER's own "
                         "(needs vda.preprocess.body_motion's output)")
    ap.add_argument("--body", help="--motion driver: SAM3D pose npz (default cache/driving/<name>/body_sam3d.npz)")
    ap.add_argument("--canonical_body", help="--motion driver: canonical person's body/pose file (default <canonical>_canonical_body.npz)")
    ap.add_argument("--mhr", help="--motion driver: SAM3D's mhr_model.pt (default: the path body_motion used)")
    ap.add_argument("--motion_sigma", type=float, default=2.0,
                    help="--motion driver: temporal Gaussian on the SAM3D pose, in frames (0 = raw)")
    ap.add_argument("--fps", type=float, default=24, help="resample the video to this frame rate (output fps)")
    ap.add_argument("--min_conf", type=float, default=None,
                    help="landmark-detection confidence threshold (default 0.99); lower it for stylised drivers")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = paths.cache_dir() / "driving" / a.name
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    if a.force or not frames_dir.exists():
        extract_frames(a.video, frames_dir, a.fps)
    if a.force or not (out / "codes_raw.npy").exists():
        from .xnemo_codes import XNemoCodeExtractor, read_frames
        ex = XNemoCodeExtractor()
        codes = np.stack([ex.encode(f) for f in read_frames(str(frames_dir))])
        np.save(out / "codes_raw.npy", codes)
        print(f"[vda] X-NeMo codes {codes.shape} ({ex.n_fallback} frames without a face)", flush=True)
    if a.force or not (out / "mono_flame.npz").exists():
        mono = track_mono(frames_dir, a.name, out / "p3dmm", a.min_conf)
        np.savez(out / "mono_flame.npz", **mono)
    mono = dict(np.load(out / "mono_flame.npz"))
    codes = np.load(out / "codes_raw.npy")
    n = len(sorted(frames_dir.glob("*.jpg")))
    if len(mono["frames"]) != n:
        # the tracker drops frames it could not crop; keep the tracked ones and their codes
        print(f"[vda] tracker kept {len(mono['frames'])}/{n} frames", flush=True)
        codes = codes[mono["frames"]]
    a.canonical = paths.driving_state(a.canonical)
    canon = dict(np.load(a.canonical))
    idx = canonical_motion(canon, a.take, len(codes), "static" if a.motion == "driver" else a.motion)
    st = build_state(canon, idx, mono, codes)
    if a.motion == "driver":
        bpath = Path(a.body) if a.body else out / "body_sam3d.npz"
        if not bpath.exists():
            raise SystemExit(f"--motion driver needs {bpath}: run (SAM 3D Body env)\n"
                             f"  python -m vda.preprocess.body_motion --name {a.name} --sam3d /path/to/sam-3d-body")
        body = dict(np.load(bpath))
        if len(body["ok"]) != n:
            raise SystemExit(f"{bpath} has {len(body['ok'])} frames, {frames_dir} has {n} (re-run body_motion)")
        ppath = Path(a.canonical_body) if a.canonical_body else default_canonical_body(a.canonical)
        if not ppath.exists():
            raise SystemExit(f"--motion driver needs the canonical person's body/pose file {ppath}")
        cbody = dict(np.load(ppath))
        mhr = a.mhr or str(body["mhr_path"])
        if not Path(mhr).exists():
            raise SystemExit(f"MHR model {mhr} not found: pass --mhr <sam-3d-body>/checkpoints/"
                             f"sam-3d-body-dinov3/assets/mhr_model.pt")
        mono_f = {k: mono[k] for k in FACE_KEYS if k in mono}
        st["verts"], st["kp"], st["gtw_lm"], stats = driver_motion(body, cbody, np.asarray(mono["frames"]),
                                                                    a.motion_sigma, mhr, mono_f)
        for k in ("verts", "kp", "gtw_lm"):
            st[k] = st[k].astype(canon[k].dtype)
        print(f"[vda] driver motion: SAM3D detected {stats['detected'] * 100:.0f}% of frames; head rotation vs the "
              f"canonical base {stats['head_rot_vs_base_deg'][0]:.1f}..{stats['head_rot_vs_base_deg'][1]:.1f} deg; "
              f"frame-to-frame head step median {stats['head_step_deg_median']:.2f} p99 "
              f"{stats['head_step_deg_p99']:.2f} max {stats['head_step_deg_max']:.2f} deg "
              f"(sigma {a.motion_sigma})", flush=True)
    np.savez_compressed(out / "state.npz", **st)
    print(f"[vda] driving state ({len(codes)} frames, canonical take {a.take}, motion {a.motion}) -> {out / 'state.npz'}")


if __name__ == "__main__":
    main()
