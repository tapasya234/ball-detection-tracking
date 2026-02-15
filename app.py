from pathlib import Path
import sys
from logger import logger
import time

import cv2

from src.detector.detect_soccer_ball import detect_soccer_ball
import src.utils.banner_utils as banner_utils
import src.utils.boundary_utils as boundary_utils

import tracker.tracker_constants as tracker_constants
from tracker.tracker_creator import create_tracker

# Colours for visualisation
DETECTION_COLOUR = (255, 0, 0)
TRACKING_COLOUR = (0, 255, 0)

# Define common PATHs
FILE = Path(__file__).resolve()
ROOT = FILE.parent

# Load the YOLOv4-tiny detector
net = cv2.dnn.readNetFromDarknet(
    cfgFile=ROOT / "models" / "yolov4-tiny.cfg",
    darknetModel=ROOT / "models" / "yolov4-tiny.weights",
)

# Read input video
input_file = ROOT / "input" / "input1.mp4"
cap = cv2.VideoCapture(input_file)
if not cap.isOpened():
    logger.error("Unable to open file at %s", input_file)
    sys.exit()

#  Read first frame and find the height of the frame after adding the banner
has_frame, frame = cap.read()
if not has_frame:
    logger.error("Input video has no frames %s", input_file)
    sys.exit()

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

TRACKER_TYPE = tracker_constants.TRACKER_MOSSE
video_writer = cv2.VideoWriter(
    output_dir / f"{input_file.stem}_{TRACKER_TYPE}.mp4",
    cv2.VideoWriter.fourcc(*"mp4v"),
    input_fps,
    (frame_width, frame_height),
)

tracker = None
banner_text = None
frame_index = 0
detected_frames_count = 0
tracked_frames_count = 0

start_time = time.time()
while cap.isOpened():
    has_frame, frame = cap.read()
    if not has_frame:
        break

    frame = banner_utils.add_banner(frame)

    # If the tracker isn't initialised, use the detector
    if tracker is None:
        detected_boundary = detect_soccer_ball(frame, net, frame_width, frame_height)
        if detected_boundary is not None:
            # Initialise the tracker
            tracker = create_tracker(TRACKER_TYPE)
            tracker.init(frame, detected_boundary)
            banner_text = "DETECT"
            boundary_utils.drawRectangle(frame, detected_boundary, DETECTION_COLOUR)
            detected_frames_count += 1
    else:
        did_track, tracked_boundary = tracker.update(frame)
        if did_track:
            banner_text = "TRACK"
            boundary_utils.drawRectangle(frame, tracked_boundary, TRACKING_COLOUR)
            tracked_frames_count += 1

        else:
            banner_text = None

    if banner_text is None:
        banner_utils.add_text(
            frame,
            f"Unable to {banner_text} soccer ball!",
            location=(50, 50),
            fontColour=(0, 0, 255),
        )
        banner_text = None

    frame_index += 1
    video_writer.write(frame)

    cv2.imshow("Detection + Tracking", frame)
    if cv2.waitKey(1) == 27:
        break

elapsed_time = time.time() - start_time
fps_processed = frame_index / elapsed_time
print(
    f"Processed {frame_index} frames in {elapsed_time:.2f}s (~{fps_processed:.2f} FPS)"
)
logger.info(
    "Processed %s frames in %ss. FPS: %s", frame_index, elapsed_time, fps_processed
)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
