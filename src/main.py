import os
import time

import numpy as np
from PIL import Image, ImageDraw
from cv2 import cv2 as cv

from color import calculateLoss, getColorBar, showColorBar, getChBoxs, calculateLoss2, getTransLoss, calculateLoss3
from subtitle import drawSubtitle, processSRT
from utils import (CV2ToPIL, getTextInfoPIL, funcTime, getFont)

LOSS_DECAY_RATIO = 0.8

def workOnSingleSubEvery5Frame(cap, nowSub, font, colors, k):
    print("working on ", nowSub)
    # set to the first frame
    cap.set(1, 0)
    _, frame = cap.read()
    text = nowSub.text
    im = CV2ToPIL(frame)
    draw = ImageDraw.Draw(im)
    textWidth, textHeight = getTextInfoPIL(draw, text, font=font)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    chboxs, _ = getChBoxs(draw, text, anchor=(frame_width // 2 - textWidth // 2, frame_height - k * textHeight),
                          font=font)

    nowStatus = [np.zeros(shape=(len(colors), 3))]

    # set to the start frame
    cap.set(1, nowSub.start)
    # cnt = 0
    for i in range(nowSub.end - nowSub.start):
        ret, frame = cap.read()
        boxs = [cv.cvtColor(frame[chbox[1]: chbox[3], chbox[0]:chbox[2]], cv.COLOR_BGR2LAB) for chbox in chboxs]
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
        resStatus = calculateLoss2(frame, boxs, nowStatus[-1], colors, text, chboxs)
        nowStatus.append(resStatus)
    return nowStatus


def workOnSingleSub(cap, nowSub, font, colors, transLoss, indexes, k, lastSub, initialStatus):
    print("working on ", nowSub)
    # set to the first frame
    cap.set(1, 0)
    _, frame = cap.read()
    text = nowSub.text
    im = CV2ToPIL(frame)
    draw = ImageDraw.Draw(im)
    textWidth, textHeight = getTextInfoPIL(draw, text, font=font)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    chboxs, _ = getChBoxs(draw, text, anchor=(frame_width // 2 - textWidth // 2, frame_height - k * textHeight),
                          font=font)

    # nowStatus = [np.zeros(shape=(len(colors), 3))]
    if initialStatus is None:
        nowStatus = [np.zeros(shape=(len(colors), 3))]
    else:
        nowStatus = initialStatus
        nowStatus[:, 0] = nowStatus[:, 0] * (LOSS_DECAY_RATIO) ** (nowSub.start - lastSub.end)
        nowStatus = [nowStatus]

    # set to the start frame
    cap.set(1, nowSub.start)
    # cnt = 0
    for i in range(nowSub.end - nowSub.start):
        ret, frame = cap.read()
        boxs = [cv.cvtColor(frame[chbox[1]: chbox[3], chbox[0]:chbox[2]], cv.COLOR_BGR2LAB) for chbox in chboxs]
        resStatus = calculateLoss3(frame, boxs, nowStatus[-1], colors, text, chboxs, transLoss, indexes)
        nowStatus.append(resStatus)
    return nowStatus


def getPickedColors(resStatus, colors):
    pickedColor = []
    printList = []
    lastColor = np.argmax(resStatus[-1][:, 0])
    for i in range(len(resStatus)):
        nowColor = lastColor
        # pickedColor.append(LABColors[nowColor])
        pickedColor.append(colors[nowColor])
        lastColor = int(resStatus[len(resStatus) - i - 1][nowColor][1])
        printList.append(
            str(resStatus[len(resStatus) - i - 1][nowColor][0] - resStatus[len(resStatus) - i - 2][lastColor][
                0]) + " : " +
            str(colors[nowColor]) + " : " + str(pickedColor[-1]) + " : " +
            str(resStatus[len(resStatus) - i - 1][nowColor][2]))
    with open("loss.txt", 'w') as f:
        f.write("\n".join(printList))
    return pickedColor[::-1]


def outputSingleSub(src, cap, nowSub, status, colors, k, font):
    outputDir = './videoOutput/%s' % src
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    resultPath = '%s/%s-Subtitle-newwork-%s.mp4' % (outputDir, src, str(nowSub.start))
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    videoWriter = cv.VideoWriter(resultPath, fourcc, cap.get(cv.CAP_PROP_FPS), (frame_width, frame_height))

    resColor = iter(getPickedColors(status, colors))

    cap.set(1, nowSub.start)
    frame_id = nowSub.start - 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame_id += 1
            if frame_id >= nowSub.start and frame_id <= nowSub.end:
                # RGB
                im = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(im)
                if frame_id == nowSub.start:
                    text = nowSub.text
                    textWidth, textHeight = getTextInfoPIL(draw, text, font)
                drawSubtitle(draw, text, (frame_width // 2 - textWidth // 2, frame_height - k * textHeight), font,
                             next(resColor))
                frame = cv.cvtColor(np.asarray(im), cv.COLOR_RGB2BGR)
                videoWriter.write(frame)
            else:
                videoWriter.release()
                break
        else:
            videoWriter.release()


def findChange(cap, src, font, k):
    def getMean(b, g, r):
        return np.array([np.mean(b), np.mean(g), np.mean(r)])

    cap.set(1, 0)
    outputDir = './videoOutput/%s' % src
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    resultPath = '%s/%s-Subtitle-newwork-change.mp4' % (outputDir, src)
    if os.path.exists(resultPath):
        return
    print("find Change begin!")
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    videoWriter = cv.VideoWriter(resultPath, fourcc, cap.get(cv.CAP_PROP_FPS), (frame_width, frame_height))
    lastFrame = None
    lastMean = None

    text = "Big Change"
    fpsAll = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    last = 0
    frame_id = 0
    # flag = False
    while (cap.isOpened()):
        ret, frame = cap.read()
        if lastFrame is None:
            lastFrame = frame
            (b, g, r) = cv.split(frame)
            lastMean = getMean(b, g, r)
            continue
        if ret:
            frame_id += 1
            (b, g, r) = cv.split(frame)
            nowMean = getMean(b, g, r)
            dis = np.sqrt(np.sum((nowMean - lastMean) ** 2))
            # print(dis)
            if dis > 10 or last != 0:
                if last == 0:
                    last = fps
                else:
                    last -= 1

                im = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                text = '%s Big Change' % str(frame_id)
                draw = ImageDraw.Draw(im)
                textWidth, textHeight = getTextInfoPIL(draw, text, font)
                drawSubtitle(draw, text, (frame_width // 2 - textWidth // 2, frame_height - k * textHeight), font,
                             (255, 255, 255))

                frame = cv.cvtColor(np.asarray(im), cv.COLOR_RGB2BGR)
            else:
                im = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                text = '%s normal' % str(frame_id)
                draw = ImageDraw.Draw(im)
                textWidth, textHeight = getTextInfoPIL(draw, text, font)
                drawSubtitle(draw, text, (frame_width // 2 - textWidth // 2, frame_height - k * textHeight), font,
                             (120, 255, 255))
                frame = cv.cvtColor(np.asarray(im), cv.COLOR_RGB2BGR)

            lastFrame = frame
            lastMean = nowMean
            videoWriter.write(frame)
        else:
            videoWriter.release()
            break
    print("Find Change Done! ")


def newWork(*args):
    srcName = args[0]
    k = int(args[1])
    fontSize = int(args[2])
    src = srcName[:-4]
    colorWheel = args[3]
    videoSrc = './videoSrc/%s' % srcName
    srtSrc = './subtitle/srt/%s.srt' % src

    # process srt and load video
    cap = cv.VideoCapture(videoSrc)
    fps = cap.get(cv.CAP_PROP_FPS)
    subs = processSRT(srtSrc, fps)
    # subs = subs[4:5]
    print(subs[0].start)
    # process colors and font
    font = getFont('Consolas', fontSize)

    findChange(cap, src, font, k)

    numColors = 1000
    colors = getColorBar(colorWheel, numColors)  # Range:[(0 ~ 255), (0 ~ 255), (0 ~ 255)]
    showColorBar(colorWheel, numColors)
    # colors = colors[-30:]
    print(colors.shape)

    LABColors = cv.cvtColor(np.array(colors[np.newaxis, :], dtype=np.uint8), cv.COLOR_RGB2LAB).reshape(
        (-1, 3))  # Range:[(0 ~ 255), (0 ~ 255), (0 ~ 255)]
    RGBColors = cv.cvtColor(np.array(LABColors[np.newaxis, :], dtype=np.uint8), cv.COLOR_LAB2RGB).reshape(
        (-1, 3))  # Range:[(0 ~ 255), (0 ~ 255), (0 ~ 255)]
    transLoss, indexes = getTransLoss(LABColors)
    # get color
    # k = 6
    status = {}
    lastSub = None
    lastStatus = None
    for sub in subs:
        start = time.time()

        status[sub] = workOnSingleSub(cap, sub, font, LABColors, transLoss, indexes, k, lastSub, initialStatus=lastStatus)
        lastSub = sub
        lastStatus = status[sub][-1]
        # outputSingleSub(src, cap, sub, status[sub], colors, k, font)
        print(sub, " : ", time.time() - start)

    # output
    outputDir = './videoOutput/%s' % src
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    resultPath = '%s/%s-Subtitle-newwork-fast-%s-decay.mp4' % (outputDir, src, colorWheel)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # fourcc = cv.VideoWriter_fourcc(*"H264")
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    videoWriter = cv.VideoWriter(resultPath, fourcc, fps, (frame_width, frame_height))

    # add subtitle and output
    cap.set(1, 1)
    _, frame = cap.read()
    itr = iter(subs)
    nowSub = next(itr)
    resColor = iter(getPickedColors(status[nowSub], colors))
    text = nowSub.text

    im = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    textWidth, textHeight = getTextInfoPIL(draw, text, font)

    cap.set(1, 1)

    frame_id = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame_id += 1
            if frame_id % 100 == 0:
                print("hello ", frame_id // 100)
            if frame_id >= nowSub.start and frame_id <= nowSub.end:
                # RGB
                im = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(im)
                drawSubtitle(draw, text, (frame_width // 2 - textWidth // 2, frame_height - k * textHeight), font,
                             next(resColor))
                frame = cv.cvtColor(np.asarray(im), cv.COLOR_RGB2BGR)
            elif frame_id > nowSub.end:
                try:
                    nowSub = next(itr)
                    resColor = iter(getPickedColors(status[nowSub], colors))
                    text = nowSub.text
                    im = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(im)
                    textWidth, textHeight = getTextInfoPIL(draw, text, font)
                except StopIteration:
                    nowSub.start = nowSub.end = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) + 1

            videoWriter.write(frame)
        else:
            videoWriter.release()
            break
    cap.release()

    return 0


def work(*args):
    # video src
    srcName = args[0]
    k = int(args[1])
    fontSize = int(args[2])

    src = srcName[:-4]
    videoSrc = "./videoSrc/%s" % srcName
    outputDir = "./videoOutput/%s" % src
    srtSrc = './subtitle/srt/%s.srt' % src
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    result_video = "%s/%s-Subtitle-work.mp4" % (outputDir, src)
    cap = cv.VideoCapture(videoSrc)
    print("read done")
    # video FPS
    fps_all = 1231234214124123
    # fps_all = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps_video = cap.get(cv.CAP_PROP_FPS)
    # video format
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # frame width and height
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    subs = processSRT(srtSrc, fps_video)

    frame_id = 0
    print("transfering %s" % srcName)
    print(frame_width, " ", frame_height)
    # position of the subtitle
    # k = 6

    font = getFont('Consolas', fontSize)

    numColors = 256
    colors = getColorBar('inferno', numColors)
    showColorBar("inferno", numColors)
    print(colors.shape)
    start = time.time()

    LABColors = cv.cvtColor(np.array(colors[np.newaxis, :], dtype=np.uint8), cv.COLOR_RGB2BGR).reshape((-1, 3))

    itr = iter(subs)
    nowSub = next(itr)
    status = {}
    # round #1, calculate colors of each frameork(*args):
    nowStatus = None
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_id += 1
            if frame_id % 100 == 0:
                print("hello", frame_id // 100)
            if frame_id >= nowSub.start and frame_id <= nowSub.end:
                if not nowSub in status:
                    assert nowStatus is None
                    print("1 ", frame_id)
                    print(nowSub)
                    nowStatus = [np.zeros(shape=(numColors, 3))]
                    status[nowSub] = nowStatus

                text = nowSub.text
                # text coordinate
                imgH, imgW = frame_height, frame_width
                im = CV2ToPIL(frame)
                draw = ImageDraw.Draw(im)
                textWidth, textHeight = getTextInfoPIL(draw, text, font=font)
                # drawImageSingleText(draw, text, font='Consolas', anchor=(imgW // 2 - textWidth // 2, imgH - k * textHeight))
                resStatus = calculateLoss(draw, frame, nowStatus[-1], LABColors, text,
                                          anchor=(imgW // 2 - textWidth // 2, imgH - k * textHeight), font=font)
                nowStatus.append(resStatus)
            elif frame_id > nowSub.end:
                print("2 ", frame_id)
                try:
                    nowSub = next(itr)
                    nowStatus = None
                    print(nowSub)
                except StopIteration:
                    nowSub.start = nowSub.end = fps_all + 1
        else:
            break
    cap.release()
    print("get color done")
    itr = iter(subs)
    nowSub = next(itr)
    resColor = None
    videoWriter = cv.VideoWriter(result_video, fourcc, fps_video, (frame_width, frame_height))
    cap = cv.VideoCapture(videoSrc)
    frame_id = 0
    subCnt = 0
    flag = True
    outputF = open("colors.text", 'w')
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_id += 1
            if frame_id % 100 == 0:
                print("hello", frame_id // 100)
            if frame_id >= nowSub.start and frame_id <= nowSub.end:
                if resColor is None:
                    subCnt += 1
                    print("here ", subCnt)
                    print(nowSub)

                    resStatus = status[nowSub]
                    pickedColor = []
                    lastColor = np.argmax(resStatus[-1][:, 0])
                    outputList = [(str(nowSub)) + "\n\n\n"]

                    for i in range(len(resStatus)):
                        nowColor = lastColor
                        pickedColor.append(colors[nowColor])
                        lastColor = int(resStatus[len(resStatus) - i - 1][nowColor][1])
                        outputList.append(str(nowColor) + " : " + str(colors[nowColor]) + " : " + str(
                            resStatus[len(resStatus) - i - 1][nowColor][0]))

                    resColor = iter(pickedColor[::-1])
                    # with open("pickedColors.txt", 'w') as f:
                    outputF.write("\n".join(outputList))
                    outputF.write("\n\n")

                # text coordinate
                text = nowSub.text
                imgH, imgW = frame_height, frame_width
                # RGB
                im = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(im)
                textWidth, textHeight = getTextInfoPIL(draw, text, font)
                drawSubtitle(draw, text, (imgW // 2 - textWidth // 2, imgH - k * textHeight), font, next(resColor))
                frame = cv.cvtColor(np.asarray(im), cv.COLOR_RGB2BGR)
            elif frame_id > nowSub.end:
                nowSub = next(itr)
                resColor = None
            videoWriter.write(frame)
        else:
            videoWriter.release()
            break

    print("write video need: ", time.time() - start)
    outputF.close()

    return 0


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
    k = 1

    font = getFont('Consolas', 22)

    numColors = 500
    colors = getColorBar('inferno', numColors)
    showColorBar("inferno", numColors)
    print(colors.shape)
    # status = np.array([list(zip(np.array([0 for i in range(numColors)]), np.array([0 for i in range(numColors)])))])
    status = [np.zeros((numColors, 3))]
    # status[:,0]
    # print(status)
    print(status[0].shape)

    start = time.time()

    LABColors = cv.cvtColor(np.array(colors[np.newaxis, :], dtype=np.uint8), cv.COLOR_RGB2BGR).reshape(
        (-1, 3))  # Error !
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
            textWidth, textHeight = getTextInfoPIL(draw, text, font=font)
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
            # str(status[len(status) - i - 1][nowColor][0]) + " : " +
            str(status[len(status) - i - 1][nowColor][0] - status[len(status) - i - 2][lastColor][0]) + " : " +
            str(colors[nowColor]) + " : " + str(pickedColor[-1]) + " : " +
            str(status[len(status) - i - 1][nowColor][2]))
    #    print(lastColor, end=" ")
    # print()
    with open("loss.txt", 'w') as f:
        f.write("\n".join(lossList[::-1]))
    resColor = pickedColor[::-1]
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
            textWidth, textHeight = getTextInfoPIL(draw, text, font=font)
            drawSubtitle(draw, text, (imgW // 2 - textWidth // 2, imgH - k * textHeight), font, resColor[frame_id - 1])
            frame = cv.cvtColor(np.asarray(im), cv.COLOR_RGB2BGR)
            videoWriter.write(frame)
        else:
            videoWriter.release()
            break

    print("write video need: ", time.time() - start)

    return 0


def tempTry(*args):
    srcName = args[0]
    k = int(args[1])
    fontSize = int(args[2])
    src = srcName[:-4]
    videoSrc = './videoSrc/%s' % srcName
    srtSrc = './subtitle/srt/%s.srt' % src
    # process srt and load video
    cap = cv.VideoCapture(videoSrc)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # video = np.empty((frame_height, frame_width, 3))
    frame_id = 0
    video = None
    cap.set(1, 20)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if frame_id == 0:
                video = frame[np.newaxis, :]
            else:
                video = np.vstack((video, frame[np.newaxis, :]))
            frame_id += 1
            if frame_id == 120:
                break
        else:
            break
    cap.release()


# funcTime(newWork, 'TimeLapseSwiss.mp4', 2, 45)

# funcTime(newWork, 'rawColors.mp4', 3, 40)
# funcTime(tempTry, 'BLACKPINK-How_You_Like_That.flv', 5, 40)
# funcTime(newWork, 'BLACKPINK-How_You_Like_That.flv', 5, 40)
funcTime(newWork, 'BLACKPINK-Kill_This_Love.mp4', 2, 24, 'RdBu')
# funcTime(newWork, 'BLACKPINK-Kill_This_Love.mp4', 2, 24, 'seismic')
# funcTime(newWork, 'TWICE-What_Is_Love.mp4', 2, 24, 'RdBu')
# funcTime(newWork, 'TWICE-What_Is_Love.mp4', 2, 24, 'seismic')
# for wheel in {'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds'}:
#     funcTime(newWork, 'BLACKPINK-Kill_This_Love.mp4', 2, 24, wheel)
# funcTime(newWork, 'TWICE-What_Is_Love.mp4', 5, 40)
# funcTime(newWork, 'demo_Trim.mp4', 5, 32, 'RdBu')
# funcTime(work, 'demo_Trim.mp4', 1, 20)
# funcTime(work, 'BLACKPINK-Kill_This_Love.mp4')
# funcTime(_main, 'demo_Trim.mp4')
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
