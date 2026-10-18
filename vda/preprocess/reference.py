"""Reference image preprocessing (once per identity, cached).

  1. crop    any portrait -> 720x1280 ('9:16') with the face at the convention the model was trained /
             evaluated with: 68-landmark height = 0.347 of the frame, landmark centre (0.501, 0.498)
             (calibrated on the 9:16 reference crops the model was evaluated with; edge-padded if the crop
             leaves the image)
  2. shape   single-image pixel3dmm FLAME2020 fit (cropping -> facer segmentation -> uv/normal
             prediction -> tracking), only the identity `shape` [300] is used downstream

Outputs: cache/refs/<name>/ref_916.png, flame_shape.npy, crop.json, p3dmm/ (tracker outputs)

    python prepare_reference.py --image me.jpg --name me
    python prepare_reference.py --image me.jpg --name me --min_conf 0.25  # stylised / non-photoreal faces

Needs the `vda` env with pixel3dmm (third_party/pixel3dmm + its preprocessing install) and a GPU.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .. import paths

#: crop convention (median over the 9:16 evaluation reference crops, face_alignment 68-pt landmarks)
LM_H, CX, CY = 0.347265625, 0.5008272171020508, 0.4978285789489746
#: tracker config suffix of the single-view fit the conditioning was built with
TRACK_SUFFIX = "_nV1_noPho_uv2000.0_n1000.0"


def crop_916(src, dst):
    import cv2
    import face_alignment
    _LT = face_alignment.LandmarksType
    fa = face_alignment.FaceAlignment(getattr(_LT, "TWO_D", None) or getattr(_LT, "_2D"), device="cpu")
    im = cv2.imread(str(src))
    if im is None:
        raise SystemExit(f"cannot read {src}")
    H, W = im.shape[:2]
    p = fa.get_landmarks_from_image(im[:, :, ::-1])
    if not p:
        raise SystemExit(f"no face found in {src}")
    p = max(p, key=lambda q: np.ptp(q[:, 1]))              # largest face
    ch = np.ptp(p[:, 1]) / LM_H
    cw = ch * 720 / 1280
    x0 = p[:, 0].mean() - CX * cw
    y0 = p[:, 1].mean() - CY * ch
    P = int(np.ceil(max(0, -x0, -y0, x0 + cw - W, y0 + ch - H))) + 2
    pad = cv2.copyMakeBorder(im, P, P, P, P, cv2.BORDER_REPLICATE)
    M = np.array([[720 / cw, 0, -(x0 + P) * 720 / cw], [0, 1280 / ch, -(y0 + P) * 1280 / ch]])
    o = cv2.warpAffine(pad, M, (720, 1280), flags=cv2.INTER_AREA if cw > 720 else cv2.INTER_CUBIC)
    cv2.imwrite(str(dst), o)
    rep = dict(src=f"{W}x{H}", crop=f"{cw:.0f}x{ch:.0f}", upsample=round(720 / cw, 2),
               pad_frac=round(float(max(0, -x0, -y0, x0 + cw - W, y0 + ch - H) / ch), 3))
    if rep["pad_frac"] > 0.05:
        print(f"[vda] WARNING: {rep['pad_frac']:.0%} of the crop is edge padding (the source is too tight "
              f"around the head); repeated border pixels can show up as streaks and blobs in the generation. Prefer "
              f"a wider source photo; if the image is already a 9:16 head-and-shoulders portrait like the example "
              f"references, try --no_crop", flush=True)
    return rep


def pixel3dmm_runner(work):
    """-> run(script, *args): one pixel3dmm stage in this Python, inside the pixel3dmm code base, with its data and
    tracking folders under `work`."""
    p3dmm = paths.pixel3dmm_dir()
    env = dict(os.environ, **paths.pixel3dmm_env(work),
               PYTHONPATH=str(p3dmm / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))

    def run(script, *args):
        cmd = [sys.executable, script, *args]
        print("[vda] $", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=p3dmm, env=env, check=True)
    return run


def fit_flame_shape(png, name, work, min_conf=None):
    """Run the pixel3dmm single-image FLAME fit; returns the [300] shape."""
    work = Path(work)
    # ALWAYS start clean: pixel3dmm's stages skip themselves when they find earlier outputs, so a
    # partial run (e.g. a failed detection) would silently be reused and the fit would run without
    # landmarks (tracker log "Loss for camera 0.0000") -> a wrong identity shape.
    if work.exists():
        shutil.rmtree(work)
    src = work / "src" / name            # the tracker derives the sequence name from basename(dir)
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy(png, src / "00000.png")
    run = pixel3dmm_runner(work)
    run("scripts/run_cropping.py", "--video_or_images_path", str(src),
        *(["--min_conf", str(min_conf)] if min_conf else []))
    run("scripts/run_facer_segmentation.py", "--video_name", name)
    for t in ("normals", "uv_map"):
        run("scripts/network_inference.py", f"model.prediction_type={t}", f"video_name={name}")
    run("scripts/track.py", f"video_name={name}", "batch_size=4", "cache_on_cpu=true")
    frame = work / "tracking" / f"{name}{TRACK_SUFFIX}" / "checkpoint" / "00000.frame"
    if not frame.exists():
        raise SystemExit(f"tracker produced no {frame}")
    import torch
    fd = torch.load(frame, map_location="cpu", weights_only=False)
    return np.asarray(fd["flame"]["shape"], np.float32).reshape(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--no_crop", action="store_true", help="the image already follows the 720x1280 convention")
    ap.add_argument("--min_conf", type=float, default=None,
                    help="landmark-detection confidence threshold (default 0.99); 0.25 for statues, paintings and cartoons")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = paths.cache_dir() / "refs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    png = out / "ref_916.png"
    if a.force or not png.exists():
        if a.no_crop:
            shutil.copy(a.image, png)
            rep = {"no_crop": True}
        else:
            rep = crop_916(a.image, png)
        json.dump(rep, open(out / "crop.json", "w"), indent=1)
        print(f"[vda] reference crop -> {png} {rep}", flush=True)
    if a.force or not (out / "flame_shape.npy").exists():
        shape = fit_flame_shape(png, a.name, out / "p3dmm", a.min_conf)
        np.save(out / "flame_shape.npy", shape)
        print(f"[vda] FLAME2020 shape -> {out / 'flame_shape.npy'} (|shape| {np.linalg.norm(shape):.2f})", flush=True)
    print(f"[vda] reference '{a.name}' ready: {out}")


if __name__ == "__main__":
    main()
