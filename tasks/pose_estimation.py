"""
-------------------------------
Author: Seunghyeon Kim (winstr)
-------------------------------
"""
import os
import traceback
from typing import Tuple

import cv2

from tools.pose import PoseEstimator
from tools.pose import draw_bbox, draw_pose


WEIGHTS_DIR = "./weights"
YOLOPOSE_PT = os.path.join(WEIGHTS_DIR, "yolov7-w6-pose.pt")


def download_weight_file():
    import wget

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    url = "https://github.com/WongKinYiu/yolov7/"\
        + "releases/download/v0.1/yolov7-w6-pose.pt"
    wget.download(url, WEIGHTS_DIR)
    print(" OK.")


def visualize_pose(cap: cv2.VideoCapture,
                   estimator: PoseEstimator,
                   color: Tuple[int, int, int]=(0, 255, 0)):
    while True:
        # get frame
        ret, frame = cap.read()
        assert ret, "frame end."
        frame = cv2.resize(frame, dsize=(960, 540))
        # esimate poses
        frame, preds = estimator.estimate_pose(frame)
        # visualize poses
        for pred in preds:
            draw_bbox(frame, pred, color)
            draw_pose(frame, pred, color)
        # show result
        cv2.imshow("result", frame)
        if cv2.waitKey(1) == ord("q"):
            break


def main(video_source: str):
    # Check if weight file(.pt) exists
    if not os.path.isfile(YOLOPOSE_PT):
        download_weight_file()

    # Prepare video capture
    cap = cv2.VideoCapture(video_source)
    assert cap.isOpened(), "cap is not openend."

    # Prepare pose estimator
    estimator = PoseEstimator(YOLOPOSE_PT)

    # Do pose estimation and visualization
    try:
        visualize_pose(cap, estimator)
    except:
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        cap.release()


if __name__ == "__main__":
    video_source = "pedestrians.mp4"
    main(video_source)
