"""RAW X-NeMo expression codes [T, 512] for a driving video (or a folder of frames).

Per frame (identical to the extraction of the training codes):
68-point face landmarks (face_alignment, SFD detector) -> square box of 1.1x the landmark extent ->
224x224 (skimage, anti-aliased) -> [-1, 1] -> X-NeMo MotEncoder.encode_facemot -> 512-d.
No face -> centre-square crop (reported).

    python scripts/extract_xnemo_codes.py --video drv.mp4 --out cache/driving/drv/codes_raw.npy
Weights: xnemo_motion_encoder.pth from the X-NeMo authors (CC BY-NC 4.0), put it in weights/third_party/.
"""
import argparse
import glob
import os

import numpy as np

from .. import paths


class XNemoCodeExtractor:
    def __init__(self, ckpt=None, device="cuda"):
        import torch
        from face_alignment import FaceAlignment, LandmarksType
        from ..third_party.xnemo.encoder import MotEncoder_withExtra
        ckpt = ckpt or str(paths.weights_dir() / "third_party" / "xnemo_motion_encoder.pth")
        self.model = MotEncoder_withExtra()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        self.model.to(device).eval()
        _T = getattr(LandmarksType, "TWO_D", None) or getattr(LandmarksType, "_2D")
        self.fa = FaceAlignment(_T, device=device)
        self.device = device
        self.n_fallback = 0

    def crop(self, image):
        """HxWx3 uint8 RGB -> float [224,224,3] in [0,1]."""
        from skimage.transform import resize as sk_resize
        preds = self.fa.get_landmarks(image)
        if preds is not None:
            k = preds[0]
            left, right, top, bottom = float(k[:, 0].min()), float(k[:, 0].max()), float(k[:, 1].min()), float(k[:, 1].max())
            cx, cy = (left + right) / 2, (top + bottom) / 2
            half = max(right - left, bottom - top) / 2 * 1.1
            x0, x1 = max(0, int(cx - half)), min(image.shape[1], int(cx + half))
            y0, y1 = max(0, int(cy - half)), min(image.shape[0], int(cy + half))
            crop = image[y0:y1, x0:x1]
        else:
            self.n_fallback += 1
            h, w = image.shape[:2]
            s = min(h, w)
            y0, x0 = (h - s) // 2, (w - s) // 2
            crop = image[y0:y0 + s, x0:x0 + s]
        return sk_resize(crop, (224, 224), anti_aliasing=True)

    def encode(self, image):
        import torch
        img = self.crop(image)
        t = torch.from_numpy(img.transpose(2, 0, 1)).float() * 2.0 - 1.0
        with torch.no_grad():
            emb = self.model.encode_facemot(t.unsqueeze(0).to(self.device))
        return emb[0].cpu().float().numpy().astype(np.float32)


def read_frames(path):
    import imageio.v2 as iio
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.jpg")))
        for f in files:
            yield np.asarray(iio.imread(f))[..., :3]
    else:
        for f in iio.get_reader(path):
            yield np.asarray(f)[..., :3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="mp4 or a directory of frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    ex = XNemoCodeExtractor(a.ckpt)
    codes = np.stack([ex.encode(f) for f in read_frames(a.video)])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, codes)
    print(f"[vda] {len(codes)} X-NeMo codes (|z| {np.linalg.norm(codes, axis=1).mean():.1f}, "
          f"{ex.n_fallback} frames without a detected face) -> {a.out}")


if __name__ == "__main__":
    main()
