import cv2
import numpy as np
from typing import Tuple

# The constants used when creating a blob
OBJECTNESS_THRESHOLD = 0.5
CONFIDENCE_THREHSOLD = 0.7
NMS_THRESHOLD = 0.4
MEAN = [0, 0, 0]
SCALE_FACTOR = 1 / 255
INPUT_DIMENSION = 416
INPUT_SIZE = (INPUT_DIMENSION, INPUT_DIMENSION)

# ClassID of the "sports ball" which will detect the soccer ball in the video.
CLASSID_SPORTS_BALL = 32


def get_output_layer_names(net: cv2.dnn.Net):
    """
    getOutputLayerNames gets the names of the output layers which is
    important when using DarNet models.

    :param net: The DNN on which the output layers names need to be retrieved.
    :type net: cv2.dnn.Net
    """
    layerNames = net.getLayerNames()

    # Return the names of the unconnected outputs
    return [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]


def detect_soccer_ball(
    frame: cv2.typing.MatLike,
    net: cv2.dnn.Net,
    frameWidth: int,
    frameHeight: int,
) -> None | Tuple[int, int, int, int]:
    """
    detect_soccer_ball will detect soccer/sports balls in the provided frame.
    If a soccer ball is found, returns the boundary box dimensions of the soccer ball.
    If a soccer ball is not found, `None` is returned.

    :param frame: The frame on which the soccer ball is detected.
    :type frame: cv2.typing.MatLike
    :param net: The DNN used for soccer ball detection.
    :type net: cv2.dnn.Net
    :param frameWidth: The width of the frame.
    :type frameWidth: int
    :param frameHeight: The height of the frame.
    :type frameHeight: int
    """

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=SCALE_FACTOR,
        size=INPUT_SIZE,
        mean=MEAN,
        swapRB=True,
        crop=False,
    )

    net.setInput(blob)
    outputs = net.forward(get_output_layer_names(net))

    boxes = []
    confidenceList = []
    for output in outputs:
        for detection in output:
            if detection[4] > OBJECTNESS_THRESHOLD:
                scores = detection[5:]
                classID = np.argmax(scores)

                if classID == CLASSID_SPORTS_BALL:
                    centerX = int(detection[0] * frameWidth)
                    centerY = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)

                    left = int(centerX - width / 2)
                    top = int(centerY - height / 2)
                    boxes.append([left, top, width, height])

                    confidenceList.append(float(scores[classID]))

    if len(boxes) == 0:
        return None

    if len(boxes) == 1:
        return (left, top, width, height)

    indices = cv2.dnn.NMSBoxes(
        boxes, confidenceList, CONFIDENCE_THREHSOLD, NMS_THRESHOLD
    )
    if len(indices) == 0:
        return None

    return tuple(boxes[indices[0]])
