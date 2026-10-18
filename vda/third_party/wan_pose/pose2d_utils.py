# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Trimmed for video-diffusion-avatars: only the code reached by vda/preprocess/skeleton.py is kept
# (the AAPoseMeta class; vda only calls AAPoseMeta.from_kps_body); every kept line is verbatim from the Wan2.2-Animate
# preprocessing code (Apache-2.0, LICENSE_Wan2.2.txt).
import numpy as np
from typing import List


class AAPoseMeta:
    def __init__(self, meta=None, kp2ds=None):
        self.image_id = ""
        self.height = 0
        self.width = 0

        self.kps_body: np.ndarray = None
        self.kps_lhand: np.ndarray = None
        self.kps_rhand: np.ndarray = None
        self.kps_face: np.ndarray = None
        self.kps_body_p: np.ndarray = None
        self.kps_lhand_p: np.ndarray = None
        self.kps_rhand_p: np.ndarray = None
        self.kps_face_p: np.ndarray = None


        if meta is not None:
            self.load_from_meta(meta)
        elif kp2ds is not None:
            self.load_from_kp2ds(kp2ds)

    @staticmethod
    def from_kps_body(kps_body: np.ndarray, height: int, width: int):

        pose_meta = AAPoseMeta()
        pose_meta.kps_body = kps_body[:, :2]
        pose_meta.kps_body_p = kps_body[:, 2]
        pose_meta.height = height
        pose_meta.width = width
        return pose_meta

    def load_from_meta(self, meta, norm_body=True, norm_hand=False):
        
        self.image_id = meta.get("image_id", "00000.png")
        self.height = meta["height"]
        self.width = meta["width"]
        kps_body_p = []
        kps_body = []
        for kp in meta["keypoints_body"]:
            if kp is None:
                kps_body.append([0, 0])
                kps_body_p.append(0)
            else:
                kps_body.append(kp)
                kps_body_p.append(1)

        self.kps_body = np.array(kps_body)
        self.kps_body[:, 0] *= self.width
        self.kps_body[:, 1] *= self.height
        self.kps_body_p = np.array(kps_body_p)

        self.kps_lhand = np.array(meta["keypoints_left_hand"])[:, :2]
        self.kps_lhand_p = np.array(meta["keypoints_left_hand"])[:, 2]
        self.kps_rhand = np.array(meta["keypoints_right_hand"])[:, :2]
        self.kps_rhand_p = np.array(meta["keypoints_right_hand"])[:, 2]

    @staticmethod
    def load_from_kp2ds(kp2ds: List[np.ndarray], width: int, height: int): 
        """input 133x3 numpy keypoints and output AAPoseMeta

        Args:
            kp2ds (List[np.ndarray]): _description_
            width (int): _description_
            height (int): _description_

        Returns:
            _type_: _description_
        """
        pose_meta = AAPoseMeta()
        pose_meta.width = width
        pose_meta.height = height
        kps_body = (kp2ds[[0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 20]] + kp2ds[[0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 21]]) / 2
        kps_lhand = kp2ds[91:112]
        kps_rhand = kp2ds[112:133]
        kps_face = np.concatenate([kp2ds[23:23+68], kp2ds[1:3]], axis=0)
        pose_meta.kps_body = kps_body[:, :2]
        pose_meta.kps_body_p = kps_body[:, 2]
        pose_meta.kps_lhand = kps_lhand[:, :2]
        pose_meta.kps_lhand_p = kps_lhand[:, 2]
        pose_meta.kps_rhand = kps_rhand[:, :2]
        pose_meta.kps_rhand_p = kps_rhand[:, 2]
        pose_meta.kps_face = kps_face[:, :2]
        pose_meta.kps_face_p = kps_face[:, 2]
        return pose_meta
