"""
-------------------------------
Author: Seunghyeon Kim (winstr)
-------------------------------
...
-------------------------------
...
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


class PoseEstimator():
    def __init__(self,
                 weight_path: str,
                 device_type: str=None,
                 stride: int=64,
                 cnf_th: float=0.25,
                 iou_th: float=0.65,):
        """
        params:
            weight_path (str): .pt file path
            device_type (str): e.g. "cpu", "cuda", etc.
            stride (int)     : ...
            cnf_th (float)   : confidence threshold
            iou_th (float)   : IoU threshold
        """
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
