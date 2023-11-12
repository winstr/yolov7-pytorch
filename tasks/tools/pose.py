"""
-------------------------------
Author: Seunghyeon Kim (winstr)
-------------------------------
...
-------------------------------
...
"""
from typing import Iterable

import cv2
import numpy as np
import torch
from torchvision import transforms

from yolov7.utils.general import non_max_suppression_kpt
from yolov7.utils.plots import output_to_keypoint


class PoseEstimator():
    def __init__(self,
                 weight_path: str,       # '/path/of/yolov7-w6-pose.pt'
                 device_type: str=None,  # 'cuda', 'cpu', etc.
                 stride: int=64,
                 cnf_th: float=0.25,
                 iou_th: float=0.65,):

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

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
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
        return preds

    def _to_list(self, preds: np.ndarray) -> Iterable:
        def cxcywh2xyxy(cx, cy, w, h):
            x_min = int(cx - w / 2)
            x_max = int(cx + w / 2)
            y_min = int(cy - h / 2)
            y_max = int(cy + h / 2)
            return x_min, y_min, x_max, y_max

        poses = []
        for pred in preds:
            cx, cy, w, h, box_conf = pred[2:7]
            x0, y0, x1, y1 = cxcywh2xyxy(cx, cy, w, h)
            kpts = pred[7:].tolist()
            pose = x0, y0, x1, y1, box_conf, kpts
            poses.append(pose)
        return poses

    def estimate_pose(self, img: np.ndarray):
        img = self._resize(img)
        tsr = self._to_tensor(img)
        preds = self._estimate(tsr)
        poses = self._to_list(preds)
        return poses