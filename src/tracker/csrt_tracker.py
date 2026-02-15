import cv2
from src.tracker.base_tracker import Base_Tracker


class CSRT_Tracker(Base_Tracker):
    def __init__(self):
        self.tracker = None

    def init(self, frame, boundary):
        self.tracker = cv2.legacy.TrackerCSRT().create()
        self.tracker.init(frame, tuple(boundary))

    def update(self, frame):
        return self.tracker.update(frame)
