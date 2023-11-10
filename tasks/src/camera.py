"""
-------------------------------
Author: Seunghyeon Kim (winstr)
-------------------------------
* Exception Classes:
    CameraCaptureError(Exception)
    OpenError(CameraCaptureError)
    ReadError(CameraCaptureError)
    NotGrabbedError(CameraCaptureError)
    RetrieveError(CameraCaptureError)

* Data Classes:
    CaptureMode()

* Classes:
    CameraCapture(cv2.VideoCapture)
    CameraCaptureHandler()
-------------------------------
Example:
    cap = CameraCapture('/dev/video0', 640, 480, 45)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, 100)

    proc_fn = lambda x: cv2.cvtColor(cv2.split(x)[-1],
                                     cv2.COLOR_BAYER_GB2BGR)

    cap_handler = CameraCaptureHandler(cap)
    cap_handler.display(proc_fn)
    cap_handler.release()
"""


from typing import Callable
from traceback import print_exc
from dataclasses import dataclass

import cv2
import numpy as np


class CameraCaptureError(Exception):
    def __init__(self, cap_source: str, msg: str):
        msg += f": {cap_source}"
        super().__init__(msg)


class OpenError(CameraCaptureError):
    def __init__(self, cap_source: str):
        msg = f"Failed to open"
        super().__init__(cap_source, msg)


class ReadError(CameraCaptureError):
    def __init__(self, cap_source: str):
        msg = f"Failed to read frame"
        super().__init__(cap_source, msg)


class NotGrabbedError(CameraCaptureError):
    def __init__(self, cap_source: str):
        msg = f"No frame has been grabbed"
        super().__init__(cap_source, msg)


class RetrieveError(CameraCaptureError):
    def __init__(self, cap_source: str):
        msg = f"Failed to retrieve frame"
        super().__init__(cap_source, msg)


@dataclass
class CaptureMode():
    width: int
    height: int
    fps: int


class CameraCapture(cv2.VideoCapture):
    def __init__(self, cap_source: str, width: int, height: int, fps: int):
        super().__init__(cap_source)
        self.source = cap_source
        self.mode = CaptureMode(width, height, fps)

        self.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.set(cv2.CAP_PROP_FPS, fps)


class CameraCaptureHandler():
    def __init__(self, cap: CameraCapture):
        self._cap = cap
        self._has_grabbed = False

    def __del__(self):
        self._cap.release()

    def grab(self):
        self._has_grabbed = self._cap.grab()

    def retrieve(self) -> np.ndarray:
        if not self._has_grabbed:
            raise NotGrabbedError(self._cap.source)
        self._has_grabbed = False
        ret, frame = self._cap.retrieve()
        if not ret:
            raise RetrieveError(self._cap.source)
        return frame

    def read(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret:
            raise ReadError(self._cap.source)
        return frame

    def release(self):
        if self._cap.isOpened():
            self._cap.release()

    def display(self, fn: Callable=None):
        def run():
            delay = int(1 / self._cap.mode.fps * 1000)
            while self._cap.isOpened():
                frame = self.read()
                if fn is not None:
                    frame = fn(frame)
                cv2.imshow(self._cap.source, frame)
                if cv2.waitKey(delay) == ord("q"):
                    break
        try:
            run()
        except:
            print_exc()
            self.release()
        finally:
            cv2.destroyWindow(self._cap.source)
