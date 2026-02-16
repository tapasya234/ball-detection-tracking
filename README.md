# Soccer Ball Detection & Tracking

This project implements single-object detection and tracking for soccer balls in video using YOLOv4-tiny for detection and multiple trackers (Kalman, CSRT, MOSSE) for tracking. It supports evaluation mode, which calculates quantitative metrics comparing detected and tracked boundaries frame-by-frame.

## Features

- Detection: YOLOv4-tiny for fast and accurate ball detection.

- Tracking: Choose from Kalman, CSRT, or MOSSE trackers.

- Evaluation Mode: Automatically detects and tracks every frame to generate metrics such as IoU and Euclidean distance.

- Visualization: Draws bounding boxes for detected (red) and tracked (green) boundaries.

- Video Output: Saves processed video with tracking overlay.

## Evaluation Metrics

Evaluation Metrics

For input1.mp4, the trackers achieved the following performance:

Tracker | Mean IoU | Mean Distance (px) | Output FPS|
--|--|--|--|
KALMAN | 0.615 | 53.8 | 48.8 |
CSRT | 0.868 |13.1 | 29.4 |
MOSSE | 0.782 | 22.4 | 44.5 |

### Interpretation

**KALMAN**: Fast and smooth, but less precise for fast-moving balls.

**CSRT**: Most accurate, but slower.

**MOSSE**: Lightweight, high FPS, moderately accurate.

Tracker selection depends on your priority:

- **High speed**: KALMAN or MOSSE

- **High accuracy**: CSRT

## Directory Structure

```bash
ball-detection-tracking/
├── input/                 # Input videos
├── output/                # Output videos with tracking overlays
├── models/                # YOLOv4-tiny config and weights
├── src/
│   ├── detector/          # YOLO detection code
│   ├── tracker/           # Tracker implementations
│   ├── processor/         # Video processing, evaluation
│   └── utils/             # Utility scripts (banner, boundary, etc.)
├── app.py                 # Main entry point
├── requirements.txt
└── README.md
```

## Notes

- Evaluation mode calculates compares frame-by-frame IoU and Euclidean distance between detected and tracked boundaries.

- FPS values may vary depending on system hardware.
