"""
-------------------------------
Author: Seunghyeon Kim (winstr)
-------------------------------
...
-------------------------------
...
"""

'''
import os

import numpy as np
import torch
from torchvision import transforms


def get_device(device_type: str=None) -> torch.device:
    if device_type is None:
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device)
    return device


def load_model(weight_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(weight_path)

    model = torch.load(weight_path, map_location=device)["model"]
    model.float().eval()
    if torch.cuda.is_available():
        model.half().to(device)
    return model


def transform(img: np.ndarray) -> torch.tensor:
    #tensor = transforms.
    pass


class PoseEstimator():
    def __init__(self,
                 weight_path: str,       # '/path/of/yolov7-w6-pose.pt'
                 device_type: str=None,  # 'cuda', 'cpu', etc.
                 stride: int=64,
                 cnf_th: float=0.25,
                 iou_th: float=0.65,):

        self._device = get_device(device_type)
        self._model = load_model(weight_path, self._device)
        self._stride = stride
        self._cnf_th = cnf_th
        self._iou_th = iou_th

    def transform(self, img: np.ndarray) -> torch.tensor:
        tensor = 1

    def estimate(self, img: np.ndarray):
        pass
'''