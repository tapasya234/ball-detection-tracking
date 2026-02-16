from typing import Tuple, List
from pathlib import Path

import math

from src.processor.process_video import process_video


def calculate_iou(
    bbox_A: Tuple[int, int, int, int], bbox_B: Tuple[int, int, int, int]
) -> float:
    xA = max(bbox_A[0], bbox_B[0])
    yA = max(bbox_A[1], bbox_B[1])
    xB = min(bbox_A[0] + bbox_A[2], bbox_B[0] + bbox_B[2])
    yB = min(bbox_A[1] + bbox_A[3], bbox_B[1] + bbox_B[3])

    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    bbox_A_area = bbox_A[2] * bbox_A[3]
    bbox_B_area = bbox_B[2] * bbox_B[3]
    return intersection_area / float(bbox_A_area + bbox_B_area - intersection_area)


def calculate_euclidean_distance(
    bbox_A: Tuple[int, int, int, int], bbox_B: Tuple[int, int, int, int]
) -> float:
    ax, ay, aw, ah = bbox_A
    bx, by, bw, bh = bbox_B
    center_A = (ax + aw / 2, ay + ah / 2)
    center_B = (bx + bw / 2, by + bh / 2)
    return math.sqrt(
        (center_A[0] - center_B[0]) ** 2 + (center_A[1] - center_B[1]) ** 2
    )


def evaluate_tracker(input_file: Path, tracker_type: str):
    result = process_video(
        input_file=input_file,
        tracker_type=tracker_type,
        should_show_live_output=False,
        is_in_evaluation_mode=True,
    )

    ious = []
    distances = []
    for detected_bbox, tracked_bbox in zip(
        result.detected_bbox_list, result.tracked_bbox_list
    ):
        if detected_bbox is None or tracked_bbox is None:
            continue

        ious.append(calculate_iou(detected_bbox, tracked_bbox))
        distances.append(calculate_euclidean_distance(detected_bbox, tracked_bbox))

    mean_iou = sum(ious) / len(ious)
    mean_distance = sum(distances) / len(distances)

    return {
        "tracker_type": result.tracker_type,
        "input_path": str(result.input_path),
        "input_fps": result.input_fps,
        "output_fps": result.frame_count / result.processing_time,
        "mean_iou": mean_iou,
        "mean_distance": mean_distance,
    }


def evaluate_all_trackers(input_file: Path, trackers_list: List[str]):
    return [
        evaluate_tracker(input_file=input_file, tracker_type=tracker)
        for tracker in trackers_list
    ]
