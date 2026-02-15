class Base_Tracker:
    def init(self, frame, boundary):
        raise NotImplementedError

    def update(self, frame):
        raise NotImplementedError
