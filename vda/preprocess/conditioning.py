"""Build the model's conditioning for (reference identity, driving sequence, camera trajectory).

Inputs
  --ref        reference directory from vda.preprocess.reference (its flame_shape.npy), or a FLAME2020 shape
               [300] .npy
  --driving    driving state .npz from the driving bank (vda.preprocess.driving), all in the canonical
               person's world: per-frame FLAME2020 face params (expression, jaw, eyes, neck, eyelids),
               head-placement landmarks, MHR body vertices + 70 keypoints, RAW X-NeMo codes, the rig
  --camera     a preset name (vda/assets/cameras) / spec .json, or an .npz with c2w [T,4,4] (+ K),
               or 'driving' = the camera path stored with the driving state
Outputs (in --out_dir)
  pose_proxy.mp4  per frame: the reference-shaped FLAME head carrying the driver's expression, rigidly
                  placed by a sim3 of its 68 landmarks onto the driver's head, composited over the body;
                  colour = 0.5 * Lambert gray + 0.5 * ray direction (head rays in the head frame, body
                  rays in the pelvis frame)
  pose_skel.mp4   AAPose skeleton (WAN-Animate's own drawer) of the projected body keypoints
  codes.npy       RAW X-NeMo codes [T, 512] (copied from the driving state)
  ref.png         the reference crop (from --ref <NAME / dir>), so python generate.py --cond <out_dir> needs nothing else
  meta.json

The render maths are those the training and evaluation conditionings were built with.
Run in the `vda` env on a CUDA GPU (nvdiffrast).
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .. import paths
from . import camera as C
from . import geometry as G
from .skeleton import TARGET_H, TARGET_W, compute_crop_transform, draw_skeleton, mhr70_to_aapose, transform_keypoints

FKEYS = ("expression", "jaw", "eyes", "neck", "eyelids")
NATIVE_W, NATIVE_H = 1100, 1604        # the rig camera's image size the intrinsics refer to
#: AAPose body joints left undrawn (everything below the shoulders): the conditioning is a portrait crop
TRIM_ZERO = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19]
FPS = 24


def load_state(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def cameras_for(arg, state, T):
    if arg in (None, "driving"):
        return np.asarray(state["c2w"], np.float64), np.asarray(state["K"], np.float64)
    if str(arg).endswith(".npz"):
        d = np.load(arg)
        c2w = np.asarray(d["c2w"], np.float64)
        if len(c2w) != T:
            raise SystemExit(f"camera npz has {len(c2w)} poses, driving has {T} frames")
        return c2w, np.asarray(d["K"] if "K" in d.files else state["K"], np.float64)
    rig = C.Rig.from_state(state)
    spec = C.load_spec(arg)
    n_spec = C.total_frames(spec)
    if n_spec < T and 77 / T < 0.9:             # the full model would show less than ~90% of the camera path
        print(f"[vda] note: the camera path ({n_spec} frames) is stretched to the driving's {T} frames, but the full "
              f"model generates only the first 77; add --max_frames 77 to see the whole camera path in one "
              f"full-model clip", flush=True)
    return C.trajectory(spec, rig, T), rig.K


def render(state, ref_shape, c2ws, K, out_dir, log_every=50):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T = len(state["codes"])
    keys = [k for k in FKEYS if f"z_{k}" in state]
    Z = {k: state[f"z_{k}"] for k in keys}
    shape = np.asarray(ref_shape, np.float32).reshape(1, -1)
    gtw_lm, verts, kp = state["gtw_lm"], state["verts"], state["kp"]

    faces20 = G.flame20().faces.cpu().numpy().astype(np.int32)
    hidx, Fh = G.head_trim(G.neutral20(shape), faces20)
    Kc = G.crop_cam(K, NATIVE_H, NATIVE_W)
    W, H = G.OUT_W, G.OUT_H
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    x0, y0, sx, sy = compute_crop_transform(NATIVE_W, NATIVE_H)
    gdst, sdst = out_dir / "pose_proxy.mp4", out_dir / "pose_skel.mp4"
    gtmp = str(gdst) + ".raw.mp4"
    gvw = cv2.VideoWriter(gtmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    bodies = []
    for i in range(T):
        c2w = c2ws[i]
        p = {k: Z[k][i] for k in keys}
        Vcan = G.neckless20(shape, p)
        Rh_, th_, sh_ = G.sim3_align(Vcan[G.LM], gtw_lm[i])
        Vhead = (sh_ * (Rh_ @ Vcan[hidx].T)).T + th_
        Vb, kp_w = verts[i], kp[i]
        gray_b, mask_b = G.render_shaded(Vb, G.B_FACES_CUT, Kc, c2w, W, H)
        gray_h, mask_h = G.render_shaded(Vhead, Fh, Kc, c2w, W, H)
        mask_h = cv2.erode(mask_h.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        mask_b = mask_b & ~mask_h
        d_w = (np.stack([xs, ys, np.ones_like(xs)], -1) @ np.linalg.inv(Kc).T) @ c2w[:3, :3].T
        dh = d_w @ Rh_
        dh /= np.linalg.norm(dh, axis=-1, keepdims=True)
        yb = kp_w[G.NECK] - 0.5 * (kp_w[G.L_HIP] + kp_w[G.R_HIP])
        yb /= np.linalg.norm(yb)
        xb = kp_w[G.L_HIP] - kp_w[G.R_HIP]
        xb -= yb * (xb @ yb)
        xb /= np.linalg.norm(xb)
        Rp = np.stack([xb, yb, np.cross(xb, yb)], 1)
        db = d_w @ Rp
        db /= np.linalg.norm(db, axis=-1, keepdims=True)
        out = np.zeros((H, W, 3), np.float32)
        gb = gray_b.astype(np.float32)[..., None]
        gh = gray_h.astype(np.float32)[..., None]
        out[mask_b] = np.clip(0.5 * gb[mask_b] + 0.5 * (db[mask_b] * .5 + .5) * 255, 0, 255)
        out[mask_h] = np.clip(0.5 * gh[mask_h] + 0.5 * (dh[mask_h] * .5 + .5) * 255, 0, 255)
        gvw.write(out.astype(np.uint8))
        w2c = np.linalg.inv(c2w)
        pc = (w2c[:3, :3] @ kp_w.T + w2c[:3, 3:4]).T
        uv = (K @ pc.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        bodies.append(mhr70_to_aapose(transform_keypoints(uv, x0, y0, sx, sy))[0])
        if log_every and i % log_every == 0:
            print(f"    frame {i}/{T} head_px={int(mask_h.sum())}", flush=True)
    gvw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", gtmp, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "15", str(gdst)], check=True)
    os.remove(gtmp)
    stmp = str(sdst) + ".raw.mp4"
    svw = cv2.VideoWriter(stmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (TARGET_W, TARGET_H))
    for b in bodies:
        conf = np.ones((20, 1), np.float32)
        conf[TRIM_ZERO] = 0.0
        frame = draw_skeleton(np.concatenate([b, conf], 1).astype(np.float32), TARGET_H, TARGET_W)
        svw.write(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
    svw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", stmp, "-c:v", "libx264",
                    "-crf", "16", "-pix_fmt", "yuv420p", str(sdst)], check=True)
    os.remove(stmp)
    codes = np.asarray(state["codes"], np.float32)
    nrm = float(np.linalg.norm(codes, axis=1).mean())
    if nrm < 20.0:
        raise SystemExit(f"REFUSE: code norm {nrm:.1f} -- not RAW X-NeMo codes (expected ~46-55)")
    np.save(out_dir / "codes.npy", codes)
    print(f"[vda] conditioning ({T} frames) -> {out_dir}", flush=True)
    return gdst, sdst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True,
                    help="reference name (prepare_reference.py --name), its cache dir (has flame_shape.npy) or a .npy shape")
    ap.add_argument("--driving", required=True,
                    help="driving sequence: nersemble8 (shipped), a name prepared with prepare_driving.py, or a state .npz")
    ap.add_argument("--camera", default="driving", help="preset name | spec .json | c2w .npz | 'driving'")
    ap.add_argument("--take", type=int, default=0,
                    help="use only take k (1-based) of a multi-take driving sequence (see its .json for labels)")
    ap.add_argument("--start", type=int, default=0, help="first driving frame (after --take)")
    ap.add_argument("--max_frames", type=int, default=0, help="use at most N driving frames (after --take/--start)")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    a.ref, a.driving = paths.reference_dir(a.ref), paths.driving_state(a.driving)
    state = load_state(a.driving)
    T0 = len(state["codes"])
    sel = np.arange(T0)
    if a.take:
        sel = np.where(state["timeline"] == f"take{a.take}")[0]
        if not len(sel):
            raise SystemExit(f"no take{a.take} in {a.driving} (takes: {sorted(set(state['timeline']))})")
    sel = sel[a.start:]
    if a.max_frames:
        sel = sel[:a.max_frames]
    if len(sel) != T0:
        for k in list(state):
            if np.ndim(state[k]) >= 1 and len(state[k]) == T0 and k not in ("rig",):
                state[k] = state[k][sel]
    T = len(state["codes"])
    shape_path = a.ref if a.ref.endswith(".npy") else os.path.join(a.ref, "flame_shape.npy")
    ref_shape = np.load(shape_path).reshape(-1)
    c2ws, K = cameras_for(a.camera, state, T)
    render(state, ref_shape, c2ws, K, a.out_dir)
    ref_png = os.path.join(a.ref, "ref_916.png")
    if os.path.isfile(ref_png):          # the folder is then self-contained: python generate.py --cond <out_dir>
        shutil.copy(ref_png, os.path.join(a.out_dir, "ref.png"))
    json.dump({"ref": a.ref, "driving": a.driving, "camera": a.camera, "n_frames": T},
              open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
