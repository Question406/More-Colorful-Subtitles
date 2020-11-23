from cv2 import cv2 as cv
# import matplotlib as mpl

import colour
import matplotlib.pyplot as plt
import numpy as np
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor

from color_conversion import *
from utils import getFont


# from matplotlib import cm
# from collections import OrderedDict


def randomPickColors(bar, k=3000):
    """
    :param colorRange: hue of the color you want to pick
    :param k: return a list of random picked colors
    :return:
    """
    return [0 for i in range(k)]


def deltaE(color1, color2):
    """
    :param color1:  openCV lab (numpy array)
    :param color2:
    :return:
    """
    C1 = LabColor(float(color1[0]) / 255 * 100, float(color1[1] - 127.0), float(color1[2] - 127.0))
    C2 = LabColor(float(color2[0]) / 255 * 100, float(color2[1] - 127.0), float(color2[2] - 127.0))
    return delta_e_cie2000(C1, C2)


def _d_textRegion2textColor(bbox, textColor):
    """
    :param bbox: text region of a single character, LAB color space
    :param textColor: color we try to put on text, LAB color space
    :return: euclidean distance between color in the bounding box and text color,
    """
    # assert len(bbox.shape) == 3
    # colorBBox = bbox.reshape((-1, 3))
    # print(np.mean(colorBBox, axis=0), end=' ')

    # return np.sqrt(np.sum((np.mean(colorBBox, axis=0) - textColor) ** 2, axis=-1))

    # retx= urn np.mean(np.sqrt(np.sum((colorBBox - textColor) ** 2, axis=-1)))
    # bbox = np.mean(colorBBox, axis=0)

    return deltaE(bbox, textColor)


def d_textRegion2textColor(bboxs, textColor):
    """
    :param bboxs: text region of all characters, LAB color space
    :param textColor: color we try to put on text, LAB color space
    :return: minimum textRegion2textColor distance:
             min(d(bbox, textColor)), where d is a distance function
    """
    l = len(bboxs)
    bboxs = [opencvLAB2standardLAB(box) for box in bboxs]
    p = opencvLAB2standardLAB(textColor)
    d = colour.delta_E(bboxs, np.tile(p, (l, 1)), method='CIE 2000')
    t = np.argsort(d)[0]
    return d[t], t
    # return d[np.argsort(d)[0]], np.argsort(d)[0]


def d_textRegion2textColor2(bboxs, textColor):
    """
    :param bboxs: text region of all characters, LAB color space
    :param textColor: color we try to put on text, LAB color space
    :return: minimum textRegion2textColor distance:
             min(d(bbox, textColor))
    """
    # bboxs = bboxs.reshape((bboxs.shape[0], -1, 3))
    # bboxsMean = np.mean(bboxs, axis=1)

    d = [_d_textRegion2textColor(bbox, textColor) for bbox in bboxs]
    # return np.min(d), np.argmin(d)
    t = np.argsort(d)[0]
    return d[t], t


def showColorBar(bar_kind, num):
    """
    :param bar_kind: color tone to choose
    :param num: the number of colors you pick from this color tone
    """
    cmap = plt.get_cmap(bar_kind)
    gradient = np.linspace(0, 1, num)
    gradient = np.vstack((gradient, gradient))  # convert to 2-dimension
    fig, ax = plt.subplots(nrows=1)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    ax.set_title(bar_kind + ' colormaps', fontsize=14)
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    plt.show()


def showColorBarWithArray(bar_array):
    """
    :param bar_array: color bar array ([R, G, B]) with scale (0, 255)
    """
    bar_array = bar_array.astype(int)
    gradient = np.stack((bar_array, bar_array), axis=0)  # convert to 2-dimension
    fig, ax = plt.subplots(nrows=1)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    ax.set_title('colormaps', fontsize=14)
    ax.imshow(gradient, aspect='auto')
    ax.set_axis_off()
    plt.show()


def getColorBar(bar_kind, num):
    """
    :param bar_kind: the color tone to choose
    :param num: the number of colors you want to pick from this color tone
    :return: the RGB color array picked from this color tone
    """
    cmap = plt.get_cmap(bar_kind)
    gradient = np.linspace(0, 1, num)
    return cmap(gradient)[:, :3] * 255


def transColorLoss(c1, c2):
    print(c1)
    print(c2)
    return np.sqrt(np.sum((c1 - c2) ** 2, -1))


def getChBoxs(image, text, anchor, font):
    lastl, u = anchor[0], anchor[1]
    chboxs = []
    textCache = []
    s = ''
    # for each character, find its bounding box
    for ch in text:
        s = s + ch
        if ch == ' ':
            continue
        else:
            box = image.textbbox((lastl, u), s, font=font)
            chboxs.append(box)
            textCache.append(s)
            lastl = box[2]
            s = ''
    return np.array(chboxs), textCache


def getTransLoss(colors):
    l = len(colors)
    tcolors = [opencvLAB2standardLAB(color) for color in colors]
    temp = [colour.delta_E(tcolors, np.tile(color, (l, 1)), method='CIE 2000') for color in tcolors]
    Temp = [t.argsort()[:35] for t in temp]
    res = np.stack(temp)
    res2 = np.stack(Temp)
    return res, res2


def getBoxMean(boxs):
    bboxs = np.stack([box.reshape(-1, 3).mean(axis=0) for box in boxs]).astype(np.float32)
    return bboxs


def calculateLoss3(frame, boxs, lastStatus, colors, text, chboxs, transLoss, indexes):
    epsilon = 20
    resStatus = np.empty(shape=(len(colors), 3), dtype='object')
    boxs = getBoxMean(boxs)
    boxs_standardLAB = opencvLAB2standardLAB(boxs)  # shape: (box_num x 3)
    palatte_standardLAB = opencvLAB2standardLAB(colors)    # shape: (256 x 3)
    distance = colour.delta_E(boxs_standardLAB[None, :, :], palatte_standardLAB[:, None, :], method='CIE 2000')  # shape: (256 x box_num)
    min_distance_per_color = distance.min(axis=1)

    # ind = indexs
    for (i, status) in enumerate(resStatus):
        ind = indexes[i]
        # indexes = transLoss[:, i].argsort()[:epsilon]
        temp = ind[int(np.argmax(lastStatus[ind, 0] - transLoss[ind, i] ** 2))]
        # temp = int(np.argmax(lastStatus[:, 0] - transLoss[:, i] ** 2))
        status[0] = lastStatus[int(temp)][0] + min_distance_per_color[i]
        status[1] = int(temp)
        status[2] = 0
    return resStatus


def calculateLoss2(frame, boxs, lastStatus, colors, text, chboxs, transLoss):
    epsilon = 10
    resStatus = np.empty(shape=(len(colors), 3), dtype='object')
    boxs = getBoxMean(boxs)

    for (i, status) in enumerate(resStatus):
        curColorLoss, charPos = d_textRegion2textColor(boxs, colors[i])
        charPos = charPos
        temp = max(i - epsilon, 0) + int(np.argmax(lastStatus[max(i - epsilon, 0): min(len(colors), i + epsilon), 0]))
        # temp = max(i - epsilon, 0) + int(np.argmax(lastStatus[max(i - epsilon, 0): min(len(colors), i + epsilon), 0] + 100 * transLoss[max(i - epsilon, 0): min(len(colors), i + epsilon), i]))
        # temp = max(i - epsilon, 0) + int(np.argmax(100 * transLoss[max(i - epsilon, 0): min(len(colors), i + epsilon), i]))
        status[0] = lastStatus[int(temp)][0] + curColorLoss
        # status[0] = lastStatus[int(temp)][0]
        status[1] = int(temp)
        status[2] = charPos

    return resStatus


def calculateLoss(image, frame, lastStatus, colors, text, anchor, font=getFont('Consolas', 32)):
    epsilon = 10
    # bounding box of entire text
    #    bbox = image.textbbox(anchor, text, font)
    # left upper bound of bounding box
    lastl, u = anchor[0], anchor[1]

    chboxs = []
    s = ''
    # for each character, find its bounding box
    for ch in text:
        s = s + ch
        if ch == ' ':
            continue
        else:
            box = image.textbbox((lastl, u), s, font=font)
            chboxs.append(box)
            lastl = box[2]
            s = ''
    boxs = [frame[chbox[1]: chbox[3], chbox[0]:chbox[2]] for chbox in chboxs]

    resStatus = np.empty(shape=(len(colors), 3), dtype='object')

    for (i, status) in enumerate(resStatus):
        curColorLoss, charPos = d_textRegion2textColor(boxs, colors[i])
        # charPos = text[charPos]
        charPos = charPos
        # colorLoss = 1e30
        status[1] = max(i - epsilon, 0) + np.argmax(lastStatus[max(i - epsilon, 0): min(len(colors), i + epsilon), 0])
        # status[1]  = np.argmin()
        colorLoss = lastStatus[int(status[1])][0]
        colorLoss += curColorLoss
        # color loss
        status[0] = colorLoss
        status[2] = charPos

    return resStatus

def findMaxDeltaEColor(labColor, iter=500):
    """
    :param labColor: lab color in standard range [(0 ~ 100), (-128 ~ 127), (-128 ~ 127)]
    :param iter: random number of iteration, the result will be more precise with larger number
    :return: the approximate color with maximum deltaE regard to labColor
    """
    maxDelta_E = 0
    maxL = None
    maxa = None
    maxb = None
    for i in range(iter):
        L = np.random.randint(0, 100)
        a = np.random.randint(-128, 127)
        b = np.random.randint(-128, 127)
        delta_E = colour.delta_E(labColor, [L, a, b], method="CIE 2000")
        if delta_E > maxDelta_E:
            maxDelta_E = delta_E
            maxL = L
            maxa = a
            maxb = b
    assert maxL is not None
    assert maxa is not None
    assert maxb is not None
    maxStandardLabColor = np.array([L,a,b])
    print("color with maximum delta E found : {}".format(maxDelta_E))
    print(maxStandardLabColor)
    maxOpencvLabColor = standardLAB2opencvLAB(maxStandardLabColor)[np.newaxis, np.newaxis, :]\
        .astype(np.uint8)
    maxRGBColor = cv.cvtColor(maxOpencvLabColor, cv.COLOR_LAB2RGB).reshape((1, 1, 3))
    plt.imshow(maxRGBColor/255.0)
    plt.show()