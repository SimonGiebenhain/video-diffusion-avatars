"""MHR-70 keypoints -> WAN-Animate AAPose body-20 skeleton, drawn with WAN's own routine
(vendored in vda/third_party/wan_pose, Apache-2.0) so the conditioning matches what the model saw."""
import numpy as np

TARGET_W, TARGET_H = 720, 1280

# AAPose body index -> MHR index (0-17 direct; the toes 18/19 are averages)
BODY_MHR = [0, 69, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 2, 1, 4, 3]
RHAND_MHR = [41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37]
LHAND_MHR = [62, 45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50, 57, 56, 55, 54, 61, 60, 59, 58]


def mhr70_to_aapose(kp2d):
    """(70, 2) MHR keypoints -> (body [20,2], lhand [21,2], rhand [21,2]) in pixels."""
    body = kp2d[BODY_MHR]
    ltoe = (kp2d[15] + kp2d[16]) / 2
    rtoe = (kp2d[18] + kp2d[19]) / 2
    body = np.concatenate([body, ltoe[None], rtoe[None]], axis=0)
    return body, kp2d[LHAND_MHR], kp2d[RHAND_MHR]


def compute_crop_transform(img_w, img_h):
    """9:16 centre-crop + resize to 720x1280: new = (old - offset) * scale."""
    new_w = int(img_h * 9 / 16)
    if new_w > img_w:
        new_h = int(img_w * 16 / 9)
        x0, y0 = 0, (img_h - new_h) // 2
        crop_w, crop_h = img_w, new_h
    else:
        x0, y0 = (img_w - new_w) // 2, 0
        crop_w, crop_h = new_w, img_h
    return x0, y0, TARGET_W / crop_w, TARGET_H / crop_h


def transform_keypoints(kp2d, x0, y0, sx, sy):
    out = kp2d.copy()
    out[:, 0] = (out[:, 0] - x0) * sx
    out[:, 1] = (out[:, 1] - y0) * sy
    return out


def draw_skeleton(kp_px, height=TARGET_H, width=TARGET_W, threshold=0.5):
    """[20,3] (x, y, conf) -> RGB uint8 [H, W, 3] skeleton on black (body only, head drawn)."""
    from ..third_party.wan_pose.pose2d_utils import AAPoseMeta
    from ..third_party.wan_pose.human_visualization import draw_aapose_by_meta_new
    kp_px = np.asarray(kp_px, dtype=np.float32)
    meta = AAPoseMeta.from_kps_body(kp_px, height, width)
    # the drawer concatenates the hand arrays before checking draw_hand: zero-confidence placeholders
    meta.kps_lhand = np.zeros((21, 2), np.float32)
    meta.kps_lhand_p = np.zeros(21, np.float32)
    meta.kps_rhand = np.zeros((21, 2), np.float32)
    meta.kps_rhand_p = np.zeros(21, np.float32)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    return draw_aapose_by_meta_new(canvas, meta, threshold=threshold, draw_hand=False, draw_head=True)
