"""
-------------------------------
Author: Seunghyeon Kim (winstr)
-------------------------------
NODES = {0: "nose", 1: "L_eye", 2: "R_eye", 3: "L_ear", 4: "R_ear",
         5: "L_shoulder", 6: "R_shoulder", 7: "L_elbow", 8: "R_elbow",
         9: "L_wrist", 10: "R_wrist", 11: "L_hip", 12: "R_hip", 13: "L_knee",
         14: "R_knee", 15: "L_ankle", 16: "R_ankle"}
COLOR = ((242, 117,  26), (106,   0, 167), (143,  13, 163), (176,  42, 143),
         (202,  70,  12), (224, 100,  97), (241, 130,  76), (252, 166,  53),
         (252, 204,  37), ( 64,  67, 135), ( 52,  94, 141), ( 41, 120, 142),
         ( 32, 143, 140), ( 34, 167, 132), ( 66, 190, 113), (121, 209,  81),
         (186, 222,  39))
-------------------------------
"""
# ↓ Temporary
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]/"yolov7"))
# ↑ Temporary
from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from yolov7.utils.general import non_max_suppression_kpt
from yolov7.utils.plots import output_to_keypoint


EDGES = {0: (1, 2), 1: (3,), 2: (4,), 3: (), 4: (), 5: (6, 7, 11),
         6: (8, 12), 7: (9,), 8: (10,), 9: (), 10: (), 11: (12, 13),
         12: (14,), 13: (15,), 14: (16,), 15: (), 16: ()}


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh
    x0, y0 = x - w / 2, y - h / 2
    x1, y1 = x + w / 2, y + h / 2
    return np.array([x0, y0, x1, y1])


class PoseEstimator():

    @staticmethod
    def visualization(img: np.ndarray, preds: np.ndarray):

        def draw_bbox(pred):
            x0, y0, x1, y1 = xywh_to_xyxy(pred[:4]).astype(int)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)

        def draw_pose(pred):
            kpts = pred[5:].reshape(-1, 3)[:, :2].astype(int)
            for i in range(len(kpts)):
                stt_kpt = kpts[i].tolist()
                cv2.circle(img, stt_kpt, 2, (0, 255, 0), -1)
                for j in EDGES[i]:
                    end_kpt = kpts[j].tolist()
                    cv2.line(img, stt_kpt, end_kpt, (0, 255, 0), 1)

        for pred in preds:
            draw_bbox(pred)
            draw_pose(pred)

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
