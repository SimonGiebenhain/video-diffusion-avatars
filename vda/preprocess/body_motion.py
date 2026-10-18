"""Per-frame MHR body pose of a monocular driving video (SAM 3D Body), for `driving --motion driver`.

Runs Meta's SAM 3D Body (https://github.com/facebookresearch/sam-3d-body) on every frame and keeps the
largest detected person's raw MHR parameters: root rotation `global_rot` [3] (MHR Euler angles, extrinsic xyz, SAM3D's camera
frame), `body_pose` [133] (the first 130 drive spine / neck / head / limbs), shape / scale / hand params,
plus the MHR keypoint regressor rows 0..69 (kp70 = map @ [verts; joints]) and the path of the MHR model,
which `vda.preprocess.driving --motion driver` uses to re-pose the canonical person's body.
Nothing is smoothed or fitted here (driving.py smooths); frames without a detection are flagged (`ok`).

    # in the SAM 3D Body env (its own install: python 3.11, torch, detectron2 / vitdet detector, ...)
    python -m vda.preprocess.body_motion --name drv --sam3d /path/to/sam-3d-body [--no_fov]
    -> cache/driving/drv/body_sam3d.npz           (frames from cache/driving/drv/frames, see driving.py)
    python -m vda.preprocess.body_motion --frames dir_of_jpgs --out body.npz --sam3d /path/to/sam-3d-body

`--sam3d` = the SAM 3D Body repository root (its `tools/` provides the person detector) with the checkpoint
at checkpoints/sam-3d-body-dinov3/{model.ckpt, assets/mhr_model.pt} (override with --ckpt / --mhr).
VRAM: ~10-11 GB peak with the MoGe2 field-of-view estimator; `--no_fov` (default focal) fits an 11 GB
RTX 2080 Ti -- the FOV only changes the camera-frame root rotation by ~1.5 deg, which the transfer
(motion relative to the clip mean) mostly cancels.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PK = ("global_rot", "body_pose_params", "hand_pose_params", "shape_params", "scale_params", "expr_params")


def _mhr_head(model):
    if hasattr(model, "head_pose"):
        return model.head_pose
    for m in model.modules():
        if hasattr(m, "keypoint_mapping") and hasattr(m, "scale_mean"):
            return m
    raise RuntimeError("no MHR head with keypoint_mapping on the SAM3D model")


def run(frames, sam3d, ckpt, mhr, no_fov=False, bbox_thr=0.3):
    import cv2
    import torch
    sys.path.insert(0, str(sam3d))                       # sam_3d_body package + tools/ (detector, fov)
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from tools.build_detector import HumanDetector
    dev = torch.device("cuda")
    t0 = time.time()
    model, cfg = load_sam_3d_body(str(ckpt), device=dev, mhr_path=str(mhr))
    fov = None
    if not no_fov:
        from tools.build_fov_estimator import FOVEstimator
        fov = FOVEstimator(name="moge2", device=dev)
    est = SAM3DBodyEstimator(model, cfg, human_detector=HumanDetector("vitdet", device=dev), fov_estimator=fov)
    t_load = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    P = {k: [] for k in PK}
    ok, bbox, cam_t, focal = [], [], [], []
    t1 = time.time()
    for i, f in enumerate(frames):
        im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        outs = est.process_one_image(im, bbox_thr=bbox_thr)
        if not outs:                                     # detector miss: whole frame as the person box
            H, W = im.shape[:2]
            outs = est.process_one_image(im, bboxes=np.array([[0, 0, W, H]], np.float32))
        if not outs:
            ok.append(False)
            bbox.append(np.zeros(4, np.float32)); cam_t.append(np.zeros(3, np.float32)); focal.append(0.0)
            for k in PK:
                P[k].append(None)
            continue
        o = max(outs, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        ok.append(True)
        bbox.append(np.asarray(o["bbox"], np.float32).reshape(4))
        cam_t.append(np.asarray(o["pred_cam_t"], np.float32).reshape(3))
        focal.append(float(np.asarray(o["focal_length"]).reshape(-1)[0]))
        for k in PK:
            P[k].append(np.asarray(o[k], np.float32).reshape(-1))
        if i % 50 == 0:
            print(f"[vda] sam3d frame {i}/{len(frames)} ({time.time() - t1:.0f}s)", flush=True)
    t_run = time.time() - t1
    ok = np.asarray(ok, bool)
    if not ok.any():
        raise SystemExit("SAM3D found no person in any frame")
    for k in PK:                                         # placeholders for missed frames (flagged by `ok`)
        ref = next(x for x in P[k] if x is not None)
        P[k] = np.stack([np.zeros_like(ref) if x is None else x for x in P[k]])
    head = _mhr_head(model)
    km = head.keypoint_mapping.detach().float().cpu().numpy()[:70]
    out = dict(frames=np.array([Path(f).name for f in frames]), ok=ok,
               global_rot=P["global_rot"], body_pose=P["body_pose_params"], hand_pose=P["hand_pose_params"],
               shape=P["shape_params"], scale=P["scale_params"], expr=P["expr_params"],
               bbox=np.stack(bbox), cam_t=np.stack(cam_t), focal=np.asarray(focal, np.float32),
               kp_map70=km.astype(np.float32), mhr_path=str(Path(mhr).resolve()),
               rot_convention="euler_xyz_extrinsic", no_fov=bool(no_fov),
               peak_vram_gb=np.float32(torch.cuda.max_memory_allocated() / 2 ** 30),
               seconds_per_frame=np.float32(t_run / len(frames)), seconds_load=np.float32(t_load),
               gpu=torch.cuda.get_device_name(0))
    print(f"[vda] SAM3D on {len(frames)} frames ({ok.mean() * 100:.0f}% detected): {t_run:.0f}s "
          f"({t_run / len(frames):.2f} s/frame) + {t_load:.0f}s load, peak VRAM (torch) "
          f"{out['peak_vram_gb']:.1f} GB on {out['gpu']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="driving name: frames from / output to cache/driving/<name>/")
    ap.add_argument("--frames", help="directory of frames (*.jpg / *.png), instead of --name")
    ap.add_argument("--out", help="output .npz (default cache/driving/<name>/body_sam3d.npz)")
    ap.add_argument("--sam3d", default=os.environ.get("SAM3D_ROOT"), help="SAM 3D Body repo root ($SAM3D_ROOT)")
    ap.add_argument("--ckpt", help="default <sam3d>/checkpoints/sam-3d-body-dinov3/model.ckpt")
    ap.add_argument("--mhr", help="default <sam3d>/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
    ap.add_argument("--no_fov", action="store_true", help="skip the MoGe2 FOV estimator (fits 11 GB GPUs)")
    ap.add_argument("--bbox_thr", type=float, default=0.3)
    a = ap.parse_args()
    if not a.sam3d:
        raise SystemExit("--sam3d (or $SAM3D_ROOT) = the SAM 3D Body repository root is required")
    if a.name:
        from .. import paths
        d = paths.cache_dir() / "driving" / a.name
        frames_dir, out = Path(a.frames or d / "frames"), Path(a.out or d / "body_sam3d.npz")
    elif a.frames and a.out:
        frames_dir, out = Path(a.frames), Path(a.out)
    else:
        raise SystemExit("give --name, or --frames and --out")
    frames = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    if not frames:
        raise SystemExit(f"no frames in {frames_dir} (run vda.preprocess.driving first, it extracts them)")
    ck = Path(a.sam3d) / "checkpoints" / "sam-3d-body-dinov3"
    res = run(frames, Path(a.sam3d), a.ckpt or ck / "model.ckpt", a.mhr or ck / "assets" / "mhr_model.pt",
              a.no_fov, a.bbox_thr)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    print(f"[vda] body motion -> {out}", flush=True)


if __name__ == "__main__":
    main()
