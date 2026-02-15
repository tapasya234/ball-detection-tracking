from pathlib import Path
from logger import logger
import time

from typing import Tuple, Optional

import cv2

from src.detector.detect_soccer_ball import detect_soccer_ball
import src.utils.banner_utils as banner_utils
import src.utils.boundary_utils as boundary_utils

import src.tracker.tracker_constants as tracker_constants
from src.tracker.tracker_creator import create_tracker

from result import Result

# Colours for visualisation
DETECTION_COLOUR = (255, 0, 0)
DETECTION_FONT_COLOUR = (255, 100, 0)
TRACKING_COLOUR = (0, 255, 0)

# Define common PATHs
FILE = Path(__file__).resolve()
ROOT = FILE.parent

# Load the YOLOv4-tiny detector
net = cv2.dnn.readNetFromDarknet(
    cfgFile=ROOT / "models" / "yolov4-tiny.cfg",
    darknetModel=ROOT / "models" / "yolov4-tiny.weights",
)


def process_video(
    input_file: Path,
    tracker_type: str = tracker_constants.TRACKER_MOSSE,
    detection_interval: int = 10,
    miss_threshold: int = 5,
    should_show_live_output: bool = False,
) -> Result:
    """
    Runs the single-object detection + tracking pipeline
    using YOLO for detection and provided tracker for tracking.

    :param input_file: Path to the input video
    :type input_file: Path
    :param tracker_type: Type of tracker to use for tracking
    :type tracker_type: str
    :param detection_interval: The frames count after which the app should re-detect
    :type detection_interval: int
    :param miss_threshold: The max number of missed tracker predictions before
    the app should re-detect the soccer ball
    :type miss_threshold: int
    :return: Returns the detection and tracking frame count along with output FPS
    :rtype: Result
    """

    # Read input video
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        logger.error("Unable to open file at %s", input_file)
        raise FileNotFoundError(f"Unable to open file {input_file}")

    #  Read first frame and find the height of the frame after adding the banner
    has_frame, frame = cap.read()
    if not has_frame:
        logger.error("Video has no frames %s", input_file)
        raise RuntimeError(f"Video has no frames: {input_file}")

    # Get details about input video that will be used for the output.
    frame = banner_utils.add_banner(frame)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = frame.shape[0]
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    logger.info(
        "Input frame count: %s fps: %s",
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FPS)),
    )

    # Make a VideoWriter to use to write the output video
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{input_file.stem}_{tracker_type}.mp4"
    video_writer = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter.fourcc(*"mp4v"),
        input_fps,
        (frame_width, frame_height),
    )

    tracker = None
    banner_text: Optional[str] = None
    frame_index = 0
    detected_frames_count = 0
    tracked_frames_count = 0
    tracking_misssed_count = 0
    miss_count = 0
    should_detect = False

    start_time = time.time()
    while cap.isOpened():
        has_frame, frame = cap.read()
        if not has_frame:
            break

        frame = banner_utils.add_banner(frame)
        should_detect = False

        if tracker is not None:
            banner_text = "TRACK"
            did_track, tracked_boundary = tracker.update(frame)
            if did_track:
                boundary_utils.draw_rectangle(frame, tracked_boundary, TRACKING_COLOUR)
                banner_utils.add_text(
                    frame,
                    f"Tracked boundary: {tracked_boundary}",
                    fontColour=TRACKING_COLOUR,
                )
                tracked_frames_count += 1
            else:
                miss_count += 1
        else:
            should_detect = True

        if (
            frame_index % detection_interval == 0
            or miss_count >= miss_threshold
            or should_detect
        ):
            tracking_misssed_count += miss_count
            miss_count = 0
            detected_boundary = detect_soccer_ball(
                frame, net, frame_width, frame_height
            )
            banner_text = "DETECT"
            if detected_boundary is not None:
                # Initialise the tracker
                tracker = create_tracker(tracker_type)
                tracker.init(frame, detected_boundary)

                boundary_utils.draw_rectangle(
                    frame, detected_boundary, DETECTION_COLOUR
                )
                banner_utils.add_text(
                    frame,
                    f"Detected boundary: {detected_boundary}",
                    fontColour=DETECTION_FONT_COLOUR,
                )
                detected_frames_count += 1

        if tracker is None and detected_boundary is None:
            banner_utils.add_text(
                frame,
                f"Unable to {banner_text} soccer ball!",
                location=(50, 50),
                fontColour=(0, 0, 255),
            )
            banner_text = None

        frame_index += 1
        video_writer.write(frame)

        if should_show_live_output:
            cv2.imshow("Detection + Tracking", frame)
            if cv2.waitKey(1) == 27:
                break

    elapsed_time = time.time() - start_time
    fps_processed = frame_index / elapsed_time
    logger.info(
        "Processed %s frames in %ss. FPS: %s", frame_index, elapsed_time, fps_processed
    )
    tracking_misssed_count += miss_count

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    return Result(
        input_path=input_file,
        output_path=output_file,
        tracker_type=tracker_type,
        input_fps=input_fps,
        frame_count=frame_index,
        processing_time=elapsed_time,
        detected_frame_count=detected_frames_count,
        tracked_frame_count=tracked_frames_count,
        tracking_missed_count=tracking_misssed_count,
    )


if __name__ == "__main__":
    input_file = ROOT / "input" / "input1.mp4"
    result = process_video(
        input_file, tracker_constants.TRACKER_CSRT, should_show_live_output=True
    )
    print(result)
