"""
-------------------------------
Author: Seunghyeon Kim (winstr)
-------------------------------
"""
import os
import traceback
from typing import Tuple

import cv2

from tools.sort import SortTracker
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


def visualize_tracked_pose(cap: cv2.VideoCapture,
                           estimator: PoseEstimator,
                           color: Tuple[int, int, int]=(0, 255, 0)):
    tracker = SortTracker()
    while True:
        # get frame
        ret, frame = cap.read()
        assert ret, "frame end."
        frame = cv2.resize(frame, dsize=(960, 540))
        # esimate poses
        frame, preds = estimator.estimate_pose(frame)

        # visualize tracked poses
        """
        if preds.ndim == 2:
            boxes = preds[:, :4]
            poses = preds[:, 5:]

            detection_map = tracker.update(boxes)
            for i in detection_map.keys():
                trk_id = detection_map[i]
                box = boxes[i]
                pose = poses[i]
                draw_tracked_pose(frame, trk_id, box, pose)
        """

        # show result
        cv2.imshow("result", frame)
        if cv2.waitKey(1) == ord("q"):
            break
"""
COLORS = (
    (242, 117, 26),
    (106, 0, 167),
    (143, 13, 163),
    (176, 42, 143),
    (202, 70, 12),
    (224, 100, 97),
    (241, 130, 76),
    (252, 166, 53),
    (252, 204, 37),
    (64, 67, 135),
    (52, 94, 141),
    (41, 120, 142),
    (32, 143, 140),
    (34, 167, 132),
    (66, 190, 113),
    (121, 209, 81),
    (186, 222, 39))

def draw_tracked_pose(mat, trk_id, bbox, pose):
    h, w, _ = mat.shape  # height, width, channel

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.2
    font_thickness = 2
    font_color = (255, 255, 255)
    
    kpts_node_radius = 2
    kpts_edge_thickness = 1
    bbox_line_thickness = 2 #kpts_edge_thickness
    color = COLORS[trk_id % len(COLORS)]

    bbox_xyxy = bbox[:4].astype(int)
    bbox_conf = round(bbox[4], 2)

    kpts_xy = pose[:, :2].astype(int)
    kpts_conf = pose[:, 2]

    # draw bounding box
    cv2.rectangle(mat, bbox_xyxy[:2], bbox_xyxy[2:], color, bbox_line_thickness)
    cv2.putText(mat, str(trk_id), bbox_xyxy[:2], font, font_scale, font_color, font_thickness)

    # draw bones
    for i in range(len(kpts_xy)):
        edges = EDGES[i]
        for j in edges:
            cv2.line(mat, kpts_xy[i], kpts_xy[j], color, kpts_edge_thickness)

    # draw joints
    for i, xy in enumerate(kpts_xy):
        cv2.circle(mat, xy, kpts_node_radius, color, -1)
"""

def main(video_source: str):
    # Check if weight file(.pt) exists
    if not os.path.isfile(YOLOPOSE_PT):
        download_weight_file()

    # Prepare video capture
    cap = cv2.VideoCapture(video_source)
    assert cap.isOpened(), "cap is not openend."

    # Prepare pose estimator
    estimator = PoseEstimator(YOLOPOSE_PT)

    # Do pose estimation, tracking and visualization
    try:
        visualize_tracked_pose(cap, estimator)
    except:
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        cap.release()


if __name__ == "__main__":
    video_source = "pedestrians.mp4"
    main(video_source)
