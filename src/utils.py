import time

import numpy as np
from PIL import Image, ImageFont
from cv2 import cv2 as cv


def getFont(font='Consolas', fontSize=32):
    """
    :param font: font name
    :param fontSize: font size
    :return: an ImgeFont object
    """
    fontPath = "fonts/%s.ttf" % font
    return ImageFont.truetype(fontPath, fontSize)


def drawImageSingleText(image, text, font=getFont('Consolas', 32), anchor=(0, 0), color=(0, 255, 0), fontSize=32):
    """
    :param image: a ImageDraw object to draw text
    :param anchor: left upper point of the text as PIL wanted
    :param text: text to draw
    :param font: font of text
    :param color: color of text
    :return: a Image object from PIL
    """
    # get font
    font = getFont(font, fontSize)
    # draw text
    image.text(anchor, text, font=font, fill=color)
    # bounding box of entire text
    bbox = image.textbbox(anchor, text, font=font)

    # draw the rectangle bounding text
    image.rectangle(bbox, fill=None, outline=color)

    # left upper bound of bbox
    lastl, u = anchor[0], anchor[1]
    image.rectangle((lastl, u, lastl, u), fill=None, outline=(255, 0, 0))

    # draw singleText box
    for ch in text:
        box = image.textbbox((lastl, u), ch, font=font)
        image.rectangle(box, fill=None, outline=(255, 0, 0))
        lastl = box[2]

    return image


def getTextInfoPIL(image, text, font='Consolas', fontSize=32):
    """
    :param image: PIl image object
    :param text: text we want to draw
    :param font: font name
    :param fontSize: font size
    :return:
    """
    font = getFont(font, fontSize)
    return image.textsize(text, font)


def CV2ToPIL(image):
    """
    :param image: numpy array (cv2 accept)
    :return: PIL image object
    """
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2LAB))


def PILToCV2(image):
    """
    :param image: PIL image object
    :return: numpy array (cv2 accpet)
    """
    return cv.cvtColor(np.asarray(image), cv.COLOR_LAB2BGR)


def funcTime(func, *args, **kwargs):
    start = time.clock()
    func(*args)
    print("elapsed: ", time.clock() - start)
