import cv2


def draw_rectangle(frame: cv2.typing.MatLike, boundaryBox: list[int], rectangleColour):
    """
    Docstring for drawRectangleAroundSoccerBall

    :param frame: The frame on which a rectangle needs to be drawn
    :type frame: cv2.typing.MatLike
    :param boundaryBox: The boundary using which the rectangle will be drawn. It needs to be in the format of (left, top, width, height).
    :type boundaryBox: list[int]
    :param rectangleColour: The colour of the rectangle
    """
    cv2.rectangle(
        frame,
        (int(boundaryBox[0]), int(boundaryBox[1])),
        (int(boundaryBox[0] + boundaryBox[2]), int(boundaryBox[1] + boundaryBox[3])),
        rectangleColour,
        3,
        cv2.LINE_AA,
    )
