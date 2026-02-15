from logger import logger

import src.tracker.tracker_constants as tracker_constants

from src.tracker.mosse_tracker import MOSSE_Tracker
from src.tracker.csrt_tracker import CSRT_Tracker
from src.tracker.base_tracker import Base_Tracker

# from trackers.kalman_tracker import KalmanTracker


def create_tracker(tracker_type: str):
    if tracker_type == tracker_constants.TRACKER_MOSSE:
        logger.info("Using tracker type %s", tracker_type)
        return MOSSE_Tracker()

    if tracker_type == tracker_constants.TRACKER_CSRT:
        logger.info("Using tracker type %s", tracker_type)
        return CSRT_Tracker()

    logger.warning("Unknown tracker type %s, defaulting to Base_Tracker", tracker_type)
    return Base_Tracker()
