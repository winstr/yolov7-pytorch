"""
-------------------------------
Author: Seunghyeon Kim (winstr)
-------------------------------
"""
# ↓ Temporary path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]/"yolov7"))
# ↑ Temporary path
from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
from yolov7.utils.general import non_max_suppression_kpt
from yolov7.utils.plots import output_to_keypoint

""" NOTE
KEYPOINT = {0: "nose",
            1: "L_eye",
            2: "R_eye",
            3: "L_ear",
            4: "R_ear",
            5: "L_shoulder",
            6: "R_shoulder",
            7: "L_elbow",
            8: "R_elbow",
            9: "L_wrist",
            10: "R_wrist",
            11: "L_hip",
            12: "R_hip",
            13: "L_knee",
            14: "R_knee",
            15: "L_ankle",
            16: "R_ankle",} """

# Keypoint Connection Map
EDGES = {0: (1, 2),     # Keypoint 0 is connected to keypoints 1 and 2.
         1: (3,),       # Keypoint 1 is connected to kepoint 3.
         2: (4,),       # Keypoint 2 is connected to kepoint 4.
         3: (),         # Keypoint 3 has no outgoing connections.
         4: (),         # Keypoint 4 has no outgoing connections.
         5: (6, 7, 11), # Keypoint 5 is connected to keypoints 6, 7, and 11.
         6: (8, 12),    # Keypoint 6 is connected to keypoints 8 and 12.
         7: (9,),       # Keypoint 7 is connected to kepoint 9.
         8: (10,),      # Keypoint 8 is connected to kepoint 10.
         9: (),         # Keypoint 9 has no outgoing connections.
         10: (),        # Keypoint 10 has no outgoing connections.
         11: (12, 13),  # Keypoint 11 is connected to keypoints 12 and 13.
         12: (14,),     # Keypoint 12 is connected to kepoint 14.
         13: (15,),     # Keypoint 13 is connected to kepoint 15.
         14: (16,),     # Keypoint 14 is connected to kepoint 16.
         15: (),        # Keypoint 15 has no outgoing connections.
         16: ()}        # Keypoint 16 has no outgoing connections.


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh
    x0, y0 = x - w / 2, y - h / 2
    x1, y1 = x + w / 2, y + h / 2
    return np.array([x0, y0, x1, y1])


def draw_bbox(img: np.ndarray, pred: np.ndarray, color:Tuple[int, int, int]):
    x0, y0, x1, y1 = xywh_to_xyxy(pred[:4]).astype(int)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)


def draw_pose(img: np.ndarray, pred: np.ndarray, color:Tuple[int, int, int]):
    keypoints = pred[5:].reshape(-1, 3)[:, :2].astype(int)
    for i in range(len(keypoints)):
        start_keypoint = keypoints[i].tolist()
        cv2.circle(img, start_keypoint, 2, color, -1)
        for j in EDGES[i]:
            end_keypoints = keypoints[j].tolist()
            cv2.line(img, start_keypoint, end_keypoints, color, 1)


class PoseEstimator():
    def __init__(self,
                 weight_path: str,       # .pt file path
                 device_type: str=None,  # e.g. "cpu", "cuda", etc.
                 stride: int=64,
                 cnf_th: float=0.25,     # confidence threshold
                 iou_th: float=0.65,):   # iou threshold

        if device_type is None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_type)

        self._model = torch.load(weight_path, self._device)["model"]
        self._model.float().eval()
        if torch.cuda.is_available():
            self._model.half().to(self._device)

        self._stride = stride
        self._cnf_th = cnf_th
        self._iou_th = iou_th

    def _resize(self, img: np.ndarray) -> np.ndarray:
        def fit(x):
            if x % self._stride:
                x = x - (x % self._stride) + self._stride
            return x
        src_h, src_w = img.shape[:2]  # height, width
        dst_h, dst_w = fit(src_h), fit(src_w)
        return cv2.resize(img, dsize=(dst_w, dst_h))

    def _tensorize(self, img: np.ndarray) -> torch.Tensor:
        tsr = transforms.ToTensor()(img)
        tsr = torch.tensor(np.array([tsr.numpy()]))
        if torch.cuda.is_available():
            tsr = tsr.half().to(self._device)
        return tsr

    def _estimate(self, tsr: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            preds, _ = self._model(tsr)
        preds = non_max_suppression_kpt(
            prediction=preds,
            conf_thres=self._cnf_th,
            iou_thres=self._iou_th,
            nc=self._model.yaml["nc"],
            kpt_label=True,)
        preds = output_to_keypoint(preds)
        preds = preds[:, 2:]
        return preds

    def estimate_pose(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img = self._resize(img)
        tsr = self._tensorize(img)
        preds = self._estimate(tsr)
        return img, preds
