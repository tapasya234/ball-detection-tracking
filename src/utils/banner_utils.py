import cv2
import numpy as np


def add_banner(frame, heightPercentage=0.08, bannerColour=(0, 0, 0)):
    """
    add_banner adds a banner to the frame at the top of it.

    :param frame: The grame/image to which the banner is added.
    :param heightPercentage: Percentage of the frame height to add as a banner.
    :param bannerColour: The colour of the banner.
    """
    bannerHeight = int(heightPercentage * frame.shape[0])
    newFrame = np.zeros(
        (bannerHeight + frame.shape[0], frame.shape[1], 3), dtype=np.uint8
    )
    if bannerColour != (0, 0, 0):
        newFrame[:bannerHeight, :, :] = bannerColour
    newFrame[bannerHeight:, :, :] = frame
    return newFrame


def add_text(
    frame,
    text,
    location=(50, 25),
    fontScale=2,
    fontThickness=2,
    fontColour=(0, 255, 255),
):
    """
    add_text adds text on the frame using the parameters provided.

    :param frame: The frame on which text is added.
    :param text: The text that is added to the frame.
    :param location: The (width, height) at which the text should originate.
    :param fontScale: The scale of the font used to apply the text.
    :param fontThickness: The thickness of the font used to apply the text.
    :param fontColour: The colour of the font used to apply the text.
    """
    cv2.putText(
        frame,
        text,
        location,
        cv2.FONT_HERSHEY_PLAIN,
        fontScale,
        fontColour,
        fontThickness,
        cv2.LINE_AA,
    )
