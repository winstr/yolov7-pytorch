"""
-------------------------------
Author: Seunghyeon Kim (winstr)
-------------------------------
...
-------------------------------
...
"""


from typing import Any

import numpy as np
import torch
from torchvision import transforms

from yolov7.utils.general import non_max_suppression_kpt
from yolov7.utils.plots import output_to_keypoint


def get_device(device_type: str=None) -> torch.device:
    if device_type is None:
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    return device


def load_model(weight_path: str, device: torch.device) -> Any:
    model = torch.load(weight_path, map_location=device)["model"]
    model.float().eval()
    if torch.cuda.is_available():
        model.half().to(device)
    return model


def preprocess(img: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = transforms.ToTensor()(img)
    tensor = torch.tensor(np.array([tensor.numpy()]))
    if torch.cuda.is_available():
        tensor = tensor.half().to(device)
    return tensor


class PoseEstimator():
    def __init__(self,
                 weight_path: str,       # '/path/of/yolov7-w6-pose.pt'
                 device_type: str=None,  # 'cuda', 'cpu', etc.
                 stride: int=64,
                 cnf_th: float=0.25,
                 iou_th: float=0.65,):

        self._device = get_device(device_type)
        self._model  = load_model(weight_path, self._device)
        self._stride = stride
        self._cnf_th = cnf_th
        self._iou_th = iou_th

    def estimate(self, img: np.ndarray):
        tensor = preprocess(img, self._device)

        with torch.no_grad():
            preds, _ = self._model(tensor)
        preds = non_max_suppression_kpt(
            prediction=preds,
            conf_thres=self._cnf_th,
            iou_thres=self._iou_th,
            nc=self._model.yaml["nc"],
            kpt_label=True,)
        preds = output_to_keypoint(preds)

        poses = []
        for pred in preds:
            x, y, w, h, conf = preds[2:7]

            x_min = int(x - w / 2)
            x_max = int(x + w / 2)
            y_min = int(y - h / 2)
            y_max = int(y + h / 2)

            kpts = pred[7:]

            pose = [x_min, y_min, x_max, y_max, conf,] + kpts.tolist()
            poses.append(pose)

        return poses