import os
import time

from PIL import Image, ImageDraw

from color import calculateLoss, getColorBar, showColorBar, getChBoxs, calculateLoss2, calculateLoss3, calculateLoss4
from subtitle import drawSubtitle, processSRT
from utils import (CV2ToPIL, getTextInfoPIL, funcTime, getFont)

from Palette import *
from color_conversion import *
from functools import reduce
from ColorTuneAnalyzer import *
from color import *


config = {}
config["tune_num"] = 3
config["hue_range"] = 0.05
config["LOSS_DECAY_RATIO"] = 0.8
config["MAX_FRAME_SKIP"] = 24



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
        nowStatus[:, 0] = nowStatus[:, 0] * (config["LOSS_DECAY_RATIO"]) ** (nowSub.start - lastSub.end)
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


def workOnSingleSub_2(cap, now_sub, palette, font, k, color_analyzer, previous_sub, log_statistics):
    tune_num = config["tune_num"]
    hue_range = config["hue_range"]
    print("working on ", now_sub)

    # locate boxes
    cap.set(1, 0)
    _, frame_image = cap.read()
    text = now_sub.text
    im = CV2ToPIL(frame_image)
    draw = ImageDraw.Draw(im)
    textWidth, textHeight = getTextInfoPIL(draw, text, font=font)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    chboxs, _ = getChBoxs(draw, text, anchor=(frame_width // 2 - textWidth // 2, frame_height - k * textHeight),
                          font=font)

    def _is_in_search_space(hues, hue_range, color_array):
        def _is_in_single_search_space(single_hue, single_hue_range, color_array):
            return np.logical_or(
                np.abs(color_array[:, 2] - single_hue) < single_hue_range,
                np.abs(color_array[:, 2] - (single_hue + 2*np.pi)) < single_hue_range,
                np.abs(color_array[:, 2] - (single_hue_range - 2*np.pi)) < single_hue_range
            )

        return reduce(np.logical_or,
                      [_is_in_single_search_space(hue, hue_range, color_array) for hue in hues])

    # set to the start frame
    cap.set(1, now_sub.start)
    sub_status = {}
    sub_status["search_color_index"] = []
    sub_status["DP_previous_index"] = []
    sub_status["DP_loss"] = []
    sub_status["max_min_distance_color"] = []
    sub_status["total_frame"] = now_sub.end - now_sub.start + 1
    for i in range(sub_status["total_frame"]):
        ret, frame_image = cap.read()
        color_tune_standardLAB = color_analyzer.analyzeImage(frame_image)
        # color_tune = standardLAB2standardLCH(color_tune[:tune_num])
        # opposite_hue = oppositeStandardLCH(color_tune)[:, 2]

        color_tune_standardLCH = standardLAB2standardLCH(color_tune_standardLAB[:])
        opposite_hue = oppositeStandardLCH(color_tune_standardLCH)[:, 2]

        # color_tune = standardLAB2standardLCH(color_tune[:max(tune_num, len(color_tune))])
        # opposite_hue = np.hstack([color_tune, oppositeStandardLCH(color_tune)])[:, 2]

        # color_tune = standardLAB2standardLCH(color_tune[:max(tune_num, len(color_tune))])
        # opposite_hue = color_tune[:, 2]

        search_color_index = np.where(_is_in_search_space(hues=opposite_hue, hue_range=hue_range,
                                                          color_array=palette.standardLCH))[0]
        search_color_index = np.hstack([palette.center_point_index, search_color_index])
        # Init DP loss for the first frame
        if i == 0:
            if previous_sub is None or now_sub.start - previous_sub.end > config["MAX_FRAME_SKIP"]:
                palette.DP_loss[:] = np.inf
                palette.DP_loss[search_color_index] = 0
            else:
                # now_sub.start - previous_sub.end <= config["MAX_FRAME_SKIP"]
                pass

        boxes = [cv.cvtColor(frame_image[chbox[1]: chbox[3], chbox[0]:chbox[2]], cv.COLOR_BGR2LAB) for chbox in chboxs]
        DP_previous_index, DP_loss = calculateLoss4(boxes=boxes, palette=palette, log_statistics=log_statistics,
                                                    search_index=search_color_index)

        # Update palette
        palette.DP_previous_index[search_color_index] = DP_previous_index  # Can be abandoned, it's useless
        palette.DP_loss[:] = np.inf  # Can be optimized using the previous search_index
        palette.DP_loss[search_color_index] = DP_loss

        sub_status["search_color_index"].append(search_color_index)
        sub_status["DP_previous_index"].append(DP_previous_index)
        sub_status["DP_loss"].append(DP_loss)
        sub_status["max_min_distance_color"].append(standardLAB2RGB(log_statistics["max_min_distance_color"][-1]))

        log_statistics["frame"].append(i + now_sub.start)
    return sub_status


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

def getPickedColor_2(sub_status, palette):
    pickedColor = []
    last_color_index = sub_status["search_color_index"][-1][np.argmin(sub_status["DP_loss"][-1])]
    for i in range(1, sub_status["total_frame"]+1):
        pickedColor.append(palette.standardRGB[last_color_index])
        tmp_index = np.where(sub_status["search_color_index"][-i] == last_color_index)[0][0]
        last_color_index = sub_status["DP_previous_index"][-i][tmp_index]
    return pickedColor[::-1]

def DP_all_frames(status, subs, palette, log_statistics):
    last_frame = np.inf
    for now_sub in subs[::-1]:
        sub_status = status[now_sub]
        sub_status["DP_color"] = []

        if last_frame - now_sub.end > config["MAX_FRAME_SKIP"]:
            last_color_index = sub_status["search_color_index"][-1][np.argmin(sub_status["DP_loss"][-1])]
        last_frame = now_sub.start

        last_loss = None
        for i in range(1, sub_status["total_frame"] + 1):
            sub_status["DP_color"].append(palette.standardRGB[last_color_index])
            log_statistics["chosen_color"].insert(0, palette.standardLAB[last_color_index])
            tmp_index = np.where(sub_status["search_color_index"][-i] == last_color_index)[0][0]
            next_last_color_index = sub_status["DP_previous_index"][-i][tmp_index]

            if i == 1:
                last_loss = sub_status["DP_loss"][-i][tmp_index]
            else:
                log_statistics["frame_loss"].insert(0, last_loss - sub_status["DP_loss"][-i][tmp_index])
                last_loss = sub_status["DP_loss"][-i][tmp_index]

            last_color_index = next_last_color_index
        log_statistics["frame_loss"].insert(0, last_loss)

        sub_status["DP_color"] = sub_status["DP_color"][::-1]



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

def outputLog(log_statistics, file_path):
    with open(file_path, "w") as f:
        f.write("frame" + ",")
        f.write("max_min_distance" + ",")
        f.write("max_min_distance_color" + ",")
        f.write("chosen_color" + ",")
        f.write("frame_loss")
        f.write("\n")
        for i in range(len(log_statistics["frame"])):
            f.write(str(log_statistics["frame"][i]) + ",")
            f.write(str(log_statistics["max_min_distance"][i]) + ",")
            f.write(str(log_statistics["max_min_distance_color"][i]) + ",")
            f.write(str(log_statistics["chosen_color"][i]) + ",")
            f.write(str(log_statistics["frame_loss"][i]))
            f.write("\n")


def newWork(*args):
    srcName = args[0]
    k = int(args[1])
    src = srcName[:-4]
    fontSize = int(args[2])
    videoSrc = './videoSrc/%s' % srcName
    srtSrc = './subtitle/srt/%s.srt' % src
    file_name = "distancestandard-60_translossbeta-0.02_DPcolor_tolerance-3-" + src

    # Record statistics
    log_statistics = {}
    log_statistics["frame"] = []
    log_statistics["max_min_distance"] = []
    log_statistics["max_min_distance_color"] = []
    log_statistics["chosen_color"] = []
    log_statistics["frame_loss"] = []

    # process srt and load video
    cap = cv.VideoCapture(videoSrc)
    fps = cap.get(cv.CAP_PROP_FPS)
    subs = processSRT(srtSrc, fps)

    # process colors and font
    font = getFont('Consolas', fontSize)

    # findChange(cap, src, font, k)
    # analysisColorTune(cap)

    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    my_palette = load_palette()
    color_tune_analyzer = ColorTuneAnalyzer(frame_width=frame_width, frame_height=frame_height)

    status = {}
    previous_sub = None
    for sub in subs:
        start = time.time()
        status[sub] = workOnSingleSub_2(cap=cap, now_sub=sub, palette=my_palette, font=font, k=k,
                                        color_analyzer=color_tune_analyzer, previous_sub=previous_sub,
                                        log_statistics=log_statistics)
        previous_sub = sub
        print(sub, " : ", time.time() - start)

    # output
    outputDir = './videoOutput/%s' % src
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    resultPath = os.path.join(outputDir, file_name + ".mp4")
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # fourcc = cv.VideoWriter_fourcc(*"H264")
    videoWriter = cv.VideoWriter(resultPath, fourcc, fps, (frame_width, frame_height))

    # DP
    DP_all_frames(status=status, subs=subs, palette=my_palette, log_statistics=log_statistics)

    # add subtitle and output
    cap.set(1, 1)
    _, frame = cap.read()
    itr = iter(subs)
    nowSub = next(itr)
    resColor = iter(status[nowSub]["DP_color"])
    # resColor = iter(status[nowSub]["max_min_distance_color"])
    text = nowSub.text

    im = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    textWidth, textHeight = getTextInfoPIL(draw, text, font)

    cap.set(1, 1)

    frame_id = 0
    while cap.isOpened():
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
                    resColor = iter(status[nowSub]["DP_color"])
                    # resColor = iter(status[nowSub]["max_min_distance_color"])
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
    outputLog(log_statistics, file_path=os.path.join(outputDir, file_name + ".csv"))
    return 0



# funcTime(newWork, 'TimeLapseSwiss.mp4', 2, 45)

# funcTime(newWork, 'rawColors.mp4', 3, 40)
# funcTime(tempTry, 'BLACKPINK-How_You_Like_That.flv', 5, 40)
# funcTime(newWork, 'BLACKPINK-How_You_Like_That.flv', 5, 40)
funcTime(newWork, 'BLACKPINK-Kill_This_Love.mp4', 2, 24, '3d')
# funcTime(newWork, 'rawColors.mp4', 2, 24, '3d')
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
