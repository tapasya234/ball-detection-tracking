import cv2
from src.tracker.base_tracker import Base_Tracker


class MOSSE_Tracker(Base_Tracker):
    def __init__(self):
        self.tracker = None

    def init(self, frame, bbox, fps):
        self.tracker = cv2.legacy.TrackerMOSSE().create()
        self.tracker.init(frame, tuple(bbox))

    def update(self, frame):
        return self.tracker.update(frame)

    def correct(self, frame, bbox, fps):
        self.init(frame, bbox, fps)
