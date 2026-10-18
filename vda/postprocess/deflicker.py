"""Block-seam deflicker for the fast (KV-streaming) model.

The student generates 4 latents (16 frames) per block; at every block boundary the global colour
steps by ~2.3x the within-block rate while the structure barely changes. At each boundary the
per-channel colour jump BEYOND the local trend is estimated on an 18x32 low-pass grid and its
correction is spread as a symmetric linear ramp over +-8 frames. Boundaries: frames 12->13, 28->29,
... (b = 12 + 16k; latent slots 1-4 = frames 0..12, then 16 frames per block).

Streams the video twice (no full-resolution video in RAM) and encodes once with libx264.

    python scripts/deflicker.py in.mp4 out.mp4 [--report]
"""
import argparse
import subprocess

import cv2
import numpy as np


def jump_offsets(L, ramp=8, first=12, period=16):
    """Low-res correction field per frame from per-frame low-res colour grids L [T, gh, gw, 3]."""
    T = len(L)
    d = np.diff(L, axis=0)
    off = np.zeros_like(L)
    for b in range(first, T - 1, period):
        nb = [i for i in list(range(b - 4, b - 1)) + list(range(b + 2, b + 5)) if 0 <= i < T - 1]
        if not nb:
            continue
        jump = d[b] - np.median(d[nb], axis=0)
        for t in range(max(0, b - ramp + 1), b + 1):
            off[t] += 0.5 * jump * (1.0 - (b - t) / ramp)
        for t in range(b + 1, min(T, b + 1 + ramp)):
            off[t] -= 0.5 * jump * (1.0 - (t - b - 1) / ramp)
    return off


def deflicker(inp, out, ramp=8, crf=15, grid=(18, 32), sigma=1.0, report=False):
    cap = cv2.VideoCapture(inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    L, H, W = [], None, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        H, W = f.shape[:2]
        x = cv2.resize(f.astype(np.float32), (grid[1], grid[0]), interpolation=cv2.INTER_AREA)
        L.append(cv2.GaussianBlur(x, (0, 0), sigma) if sigma > 0 else x)
    cap.release()
    off = jump_offsets(np.stack(L), ramp)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", out],
                          stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(inp)
    ds, dc, ps, pc = [], [], None, None
    for t in range(len(off)):
        ok, f = cap.read()
        if not ok:
            break
        up = cv2.resize(off[t], (W, H), interpolation=cv2.INTER_CUBIC)
        g = np.clip(f.astype(np.float32) + up, 0, 255).astype(np.uint8)
        ff.stdin.write(g.tobytes())
        if report:
            gs = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gc = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if ps is not None:
                ds.append(np.abs(gs - ps).mean())
                dc.append(np.abs(gc - pc).mean())
            ps, pc = gs, gc
    cap.release()
    ff.stdin.close()
    ff.wait()
    if report:
        for name, d in (("source", np.array(ds)), ("deflickered", np.array(dc))):
            bnd = [12 + 16 * k for k in range(4096) if 76 <= 12 + 16 * k < len(d) - 5]
            r = np.array([d[b] / max(np.median(np.r_[d[b - 4:b - 1], d[b + 2:b + 5]]), 1e-3) for b in bnd])
            print(f"[seams] {name:12s} median {np.median(r):.3f}  strong(>1.5) {int((r > 1.5).sum())}/{len(r)}",
                  flush=True)
    print(f"[vda] deflickered -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--ramp", type=int, default=8)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    deflicker(a.inp, a.out, a.ramp, report=a.report)


if __name__ == "__main__":
    main()
