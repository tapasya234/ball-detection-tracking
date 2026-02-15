import cv2

# import cv2.
from base_tracker import Base_Tracker


class MOSSE_Tracker(Base_Tracker):
    def __init__(self):
        self.tracker = None

    def init(self, frame, boundary):
        self.tracker = cv2.legacy.TrackerMOSSE().create()
        self.tracker.init(frame, tuple(boundary))

    def update(self, frame):
        return self.tracker.update(frame)
