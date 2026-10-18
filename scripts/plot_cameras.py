"""Plot every camera preset as an azimuth/elevation path around the canonical head (+ radius), against the
envelope of the real capture rig the model was trained on.

    python scripts/plot_cameras.py --bank weights/driving_bank/nersemble8.npz --out docs/cameras.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vda.preprocess import camera as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=0, help="resample every preset to N frames (0 = its own length)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(a.bank)
    rig = C.Rig(d["rig"], d["K"])
    az_h, el_h = np.degrees(rig.az_half), np.degrees(rig.el_half)
    presets = C.list_presets()
    n = len(presets)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), presets):
        spec = C.load_spec(name)
        T = a.frames or C.total_frames(spec)
        az_d, el_d, rad = C.angles_deg(spec, rig, T)
        ax.add_patch(plt.Rectangle((-az_h, -el_h), 2 * az_h, 2 * el_h, color="#cfe7ea", zorder=0,
                                   label="training rig envelope"))
        sc = ax.scatter(az_d, el_d, c=np.arange(T), cmap="viridis", s=10 + 30 * (rad - rad.min() + 0.01), zorder=2)
        ax.plot(az_d, el_d, color="#7a8796", lw=0.6, zorder=1)
        ax.scatter([az_d[0]], [el_d[0]], marker="o", s=70, facecolors="none", edgecolors="#c0392b", lw=1.6, zorder=3)
        ax.set_title(f"{name}  ({T} f" + (f", radius {rad.min():.2f}-{rad.max():.2f}x)" if np.ptp(rad) > 1e-6 else ")"),
                     fontsize=10)
        lim = max(90, np.abs(az_d).max() + 10)
        ax.set_xlim(-lim if lim < 200 else min(-90, az_d.min() - 10), lim if lim < 200 else az_d.max() + 10)
        ax.set_ylim(-45, 45)
        ax.set_xlabel("azimuth (deg)", fontsize=8)
        ax.set_ylabel("elevation (deg)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01).set_label("frame", fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Camera presets around the canonical head (red circle = first frame; shaded = real rig cameras, "
                 f"+-{az_h:.0f} deg az, +-{el_h:.0f} deg el)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
