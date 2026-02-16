import cv2
import numpy as np

from src.tracker.base_tracker import Base_Tracker


class Kalman_Tracker(Base_Tracker):
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)

        # This is set to 6 because of the assumption that the
        # boundary box of the ball will be square, i.e., width == height.
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)

        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            np.float32,
        )

        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-1
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1

        self.initialised = False

    def init(self, frame, bbox, fps):
        x, y, width, height = bbox
        centerX = x + width / 2
        centerY = y + height / 2

        dt = 1 / fps
        self.set_dt(dt)

        state = np.array([[centerX], [centerY], [width], [0], [0], [0]], np.float32)
        self.kf.statePre = state
        self.kf.statePost = state.copy()
        self.initialised = True

    def update(self, frame):
        if not self.initialised:
            return False, None

        prediction = self.kf.predict()
        centerX = prediction[0].item()
        centerY = prediction[1].item()
        width = prediction[2].item()
        width = max(1, int(width))

        x = int(centerX - width / 2)
        y = int(centerY - width / 2)
        return True, (x, y, width, width)

    def correct(self, frame, bbox, fps):
        x, y, width, height = bbox
        cx = x + width / 2
        cy = y + height / 2

        measurement = np.array(
            [[np.float32(cx)], [np.float32(cy)], [np.float32(width)]],
            dtype=np.float32,
        )
        self.kf.correct(measurement)

    def set_dt(self, dt):
        self.kf.transitionMatrix[0, 3] = dt
        self.kf.transitionMatrix[1, 4] = dt
        self.kf.transitionMatrix[2, 5] = dt
