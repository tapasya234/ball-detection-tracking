from logger import logger

import tracker_constants

from mosse_tracker import MOSSE_Tracker
from base_tracker import Base_Tracker

# from trackers.kalman_tracker import KalmanTracker


def create_tracker(tracker_type: str):
    if tracker_type == tracker_constants.TRACKER_MOSSE:
        logger.info("Using tracker type %s", tracker_type)
        return MOSSE_Tracker()

    logger.warning("Unknown tracker type %s, defaulting to Base_Tracker", tracker_type)
    return Base_Tracker()
