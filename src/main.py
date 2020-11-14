import os
import time

import numpy as np
from PIL import Image, ImageDraw
from cv2 import cv2 as cv

from color import calculateLoss, getColorBar, showColorBar
from subtitle import drawSubtitle
from utils import (CV2ToPIL, getTextInfoPIL, funcTime, getFont)


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
    colors = getColorBar('inferno', numColors)
    showColorBar("inferno", numColors)
    print(colors.shape)
    # status = np.array([list(zip(np.array([0 for i in range(numColors)]), np.array([0 for i in range(numColors)])))])
    status = [np.zeros((numColors, 3))]
    # status[:,0]
    # print(status)
    print(status[0].shape)

    start = time.time()

    LABColors = cv.cvtColor(np.array(colors[np.newaxis, :], dtype=np.uint8), cv.COLOR_RGB2BGR).reshape((-1, 3))
    # round #1, calculate colors of each frame
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_id += 1
            # if not frame_id == 327:
            #     continue
            if frame_id % 100 == 0:
                print("hello", frame_id // 100)
            # text coordinate
            text = 'frame_%s' % frame_id
            imgH, imgW = frame_height, frame_width
            im = CV2ToPIL(frame)
            draw = ImageDraw.Draw(im)
            textWidth, textHeight = getTextInfoPIL(draw, text)
            # drawImageSingleText(draw, text, font='Consolas', anchor=(imgW // 2 - textWidth // 2, imgH - k * textHeight))
            resStatus = calculateLoss(draw, frame, im, status[-1], LABColors, text,
                                      anchor=(imgW // 2 - textWidth // 2, imgH - k * textHeight), font=font)
            status.append(resStatus)
        else:
            break
    cap.release()

    print("get color need: ", time.time() - start)
    start = time.time()

    pickedColor = []
    lastColor = np.argmax(status[-1][:, 0])
    lossList = []
    # print(lastColor, end=" ")
    for i in range(len(status)):
        nowColor = lastColor
        # pickedColor.append(LABColors[nowColor])
        pickedColor.append(colors[nowColor])
        lastColor = int(status[len(status) - i - 1][nowColor][1])
        lossList.append(
            str(status[len(status) - i - 1][nowColor][0]) + " : " +
            str(status[len(status) - i - 1][nowColor][0] - status[len(status) - i - 2][lastColor][0]) + " : " +
            str(colors[nowColor]) + " : " + str(pickedColor[-1]) + " : " +
            str(status[len(status) - i - 1][nowColor][2]))
    #    print(lastColor, end=" ")
    # print()
    with open("loss.txt", 'w') as f:
        f.write("\n".join(lossList[::-1]))
    resColor = pickedColor[::-1]
    # print(resColor.shape)
    # print(resColor)
    # print(len(resColor))

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
            # RGB
            im = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(im)
            textWidth, textHeight = getTextInfoPIL(draw, text)
            drawSubtitle(draw, text, (imgW // 2 - textWidth // 2, imgH - k * textHeight), font, resColor[frame_id - 1])
            frame = cv.cvtColor(np.asarray(im), cv.COLOR_RGB2BGR)
            videoWriter.write(frame)
        else:
            videoWriter.release()
            break

    print("write video need: ", time.time() - start)

    return 0


funcTime(_main, 'demo.mp4')
# funcTime(_main, 'TWICE-What_Is_Love.mp4')
# funcTime(_main, 'BLACKPINK-Kill_This_Love.mp4')
# funcTime(_main, 'WesternSichuan.flv')

# outputF = open("./temp.png", 'wb')
# im = Image.open('./videoSrc/test.png')
#
# draw = ImageDraw.Draw(im)
# text = 'frame_327'
# imgH, imgW = im.height, im.width
# textWidth, textHeight = getTextInfoPIL(draw, text)
# k = 5
# color = (255,154,255)
# boxes = drawImageSingleText(draw, text, anchor=(imgW // 2 - textWidth // 2, imgH - k * textHeight),color=color)
#
# a = np.asarray(im).copy()
# boxes = [a[box[1]: box[3], box[0]:box[2]] for box in boxes]
#
# print(d_textRegion2textColor(boxes, color))
# # a = np.asarray(im).copy()
# # for box in boxes:
# #     a[box[1]: box[3], box[0]:box[2]] = (255, 0, 0)
# # im = Image.fromarray(a)
# # draw = ImageDraw.Draw(im)
#
# im.save(outputF, "PNG")
# im.close()
# outputF.close()
