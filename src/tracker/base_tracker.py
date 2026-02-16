class Base_Tracker:
    def init(self, frame, bbox, fps):
        # bbox is expected to be a tuple in the form of (x, y, width, height)
        raise NotImplementedError

    def update(self, frame):
        raise NotImplementedError

    def correct(self, frame, bbox, fps):
        raise NotImplementedError
