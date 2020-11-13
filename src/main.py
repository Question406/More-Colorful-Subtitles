import os

import numpy as np
from PIL import ImageDraw
from cv2 import cv2 as cv

from color import calculateLoss, getColorBar, showColorBar
from subtitle import drawSubtitle
from utils import (CV2ToPIL, PILToCV2, getTextInfoPIL, funcTime, getFont)


def _main(*args):
    # video src
    srcName = args[0]
    src = srcName.strip(".mp4").strip(".flv")
    videoSrc = "./videoSrc/%s" % srcName
    outputDir = "./videoOutput/%s" % src
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    result_video = "%s/%s-Subtitle.mp4" % (outputDir, src)
    cap = cv.VideoCapture(videoSrc)
    print("read done")
    # video FPS
    fps_video = cap.get(cv.CAP_PROP_FPS)
    # video format
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # frame width and height
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    frame_id = 0
    print("transfering %s" % srcName)
    print(frame_width, " ", frame_height)
    # position of the subtitle
    k = 5

    font = getFont('Consolas', 32)

    numColors = 256
    colors = getColorBar('Blues', numColors)
    showColorBar("Blues", numColors)
    print(colors.shape)
    # status = np.array([list(zip(np.array([0 for i in range(numColors)]), np.array([0 for i in range(numColors)])))])
    status = [np.zeros((numColors, 2))]
    # status[:,0]
    # print(status)
    print(status[0].shape)

    LABColors = cv.cvtColor(colors[np.newaxis, :], cv.COLOR_RGB2BGR).reshape((-1, 3))
    # round #1, calculate colors of each frame
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_id += 1
            if frame_id % 100 == 0:
                print("hello", frame_id // 100)
            # text coordinate
            text = 'frame_%s' % frame_id
            imgH, imgW = frame_height, frame_width
            im = CV2ToPIL(frame)
            draw = ImageDraw.Draw(im)
            textWidth, textHeight = getTextInfoPIL(draw, text)
            # drawImageSingleText(draw, text, font='Consolas', anchor=(imgW // 2 - textWidth // 2, imgH - k * textHeight))
            resStatus = calculateLoss(draw, frame, im, status[-1], colors, text,
                                      anchor=(imgW // 2 - textWidth // 2, imgH - k * textHeight), font=font)
            status.append(resStatus)
        else:
            break
    cap.release()

    pickedColor = []
    lastColor = np.argmin(status[-1][:, 0])
    print(lastColor, end=" ")
    for i in range(len(status)):
        pickedColor.append(colors[lastColor])
        lastColor = int(status[len(status) - i - 1][lastColor][1])
        print(lastColor, end=" ")
    print()
    resColor = pickedColor[::-1]
    print(resColor)
    print(len(resColor))

    # return
    videoWriter = cv.VideoWriter(result_video, fourcc, fps_video, (frame_width, frame_height))
    cap = cv.VideoCapture(videoSrc)
    frame_id = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_id += 1
            if frame_id % 100 == 0:
                print("hello", frame_id // 100)
            # text coordinate
            text = 'frame_%s' % frame_id
            imgH, imgW = frame_height, frame_width
            im = CV2ToPIL(frame)
            draw = ImageDraw.Draw(im)
            textWidth, textHeight = getTextInfoPIL(draw, text)
            drawSubtitle(draw, text, (imgW // 2 - textWidth // 2, imgH - k * textHeight), font, resColor[frame_id - 1])
            frame = PILToCV2(im)
            videoWriter.write(frame)
        else:
            videoWriter.release()
            break

    return 0

funcTime(_main, 'demo_Trim.mp4')
