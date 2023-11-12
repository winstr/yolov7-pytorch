import os

import cv2

from tools.pose import PoseEstimator


WEIGHTS_DIR = "./weights"
YOLOPOSE_PT = os.path.join(WEIGHTS_DIR, "yolov7-w6-pose.pt")


def download_weight():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.isfile(YOLOPOSE_PT):
        import wget
        url = "https://github.com/WongKinYiu/yolov7/"\
            + "releases/download/v0.1/yolov7-w6-pose.pt"
        wget.download(url, WEIGHTS_DIR)
        print(" OK.")


if __name__ == "__main__":
    download_weight()
    img = cv2.imread("running.jpg")
    estimator = PoseEstimator(YOLOPOSE_PT)

    poses = estimator.estimate_pose(img)
    print(poses)
