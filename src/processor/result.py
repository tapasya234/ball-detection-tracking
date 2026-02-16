from pathlib import Path
from dataclasses import dataclass


@dataclass
class Process_Result:
    input_path: Path
    output_path: Path
    tracker_type: str
    input_fps: float
    frame_count: int
    processing_time: float
    detected_frame_count: int
    tracked_frame_count: int
    tracking_missed_count: int
    detected_bbox_list: list
    tracked_bbox_list: list
