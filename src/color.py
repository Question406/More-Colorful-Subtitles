# from cv2 import cv2 as cv
# import matplotlib as mpl

import colour
import matplotlib.pyplot as plt
import numpy as np
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor

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


def getStandardLAB(x):
    y = x.astype(np.float32)
    y[0] = x[0] / 255.0 * 100
    y[1] = x[1] - 128.0
    y[2] = x[2] - 128.0
    return y


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
    bboxs = [getStandardLAB(box) for box in bboxs]
    p = getStandardLAB(textColor)
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
    tcolors = [getStandardLAB(color) for color in colors]
    temp = [colour.delta_E(tcolors, np.tile(color, (l, 1)), method='CIE 2000') for color in tcolors]
    Temp = [t.argsort()[:30] for t in temp]
    res = np.stack(temp)
    res2 = np.stack(Temp)
    return res, res2


def getBoxMean(boxs):
    bboxs = np.stack([np.mean(box.reshape(-1, 3), axis=0) for box in boxs])
    return bboxs


def calculateLoss3(frame, boxs, lastStatus, colors, text, chboxs, transLoss, indexes):
    epsilon = 20
    resStatus = np.empty(shape=(len(colors), 3), dtype='object')
    l = len(boxs)
    boxs = getBoxMean(boxs)
    tboxs = np.tile(boxs, (len(colors), 1)).reshape((-1, 3))
    bboxs = [getStandardLAB(box) for box in tboxs]
    p = [getStandardLAB(textColor) for textColor in colors]
    d = colour.delta_E(bboxs, np.tile(p, (1, l)).reshape((-1, 3)), method='CIE 2000')
    d = d.reshape((len(colors), l))

    # ind = indexs
    for (i, status) in enumerate(resStatus):
        curColorLoss = d[i].min()
        ind = indexes[i]
        # indexes = transLoss[:, i].argsort()[:epsilon]
        temp = ind[int(np.argmax(lastStatus[ind, 0] - transLoss[ind, i] ** 2))]
        # temp = int(np.argmax(lastStatus[:, 0] - transLoss[:, i] ** 2))
        status[0] = lastStatus[int(temp)][0] + curColorLoss
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
