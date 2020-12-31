from cv2 import cv2 as cv
# import matplotlib as mpl

import colour
import matplotlib.pyplot as plt
import numpy as np
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor

from color_conversion import *
from utils import getFont


deltaE_method = "CIE 2000"
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
    d = colour.delta_E(bboxs, np.tile(p, (l, 1)), method=deltaE_method)
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
    temp = [colour.delta_E(tcolors, np.tile(color, (l, 1)), method=deltaE_method) for color in tcolors]
    Temp = [t.argsort()[:30] for t in temp]
    res = np.stack(temp)
    res2 = np.stack(Temp)
    return res, res2


def getBoxMean(boxs):
    bboxs = np.stack([box.reshape(-1, 3).mean(axis=0) for box in boxs]).astype(np.float32)
    return bboxs


def calculateLoss(boxes, palette, log_statistics, search_index, key_frame, frame_mean_delta, config):
    transloss_beta = config["transloss_beta"]
    transloss_gamma = config["transloss_gamma"]
    transloss_theta = config["transloss_theta"]
    distance_gamma = config["distance_gamma"]
    distance_standard = config["distance_standard"]
    tolerance_index = config["tolerance_index"]
    boxes = getBoxMean(boxes)
    boxes_standardLAB = opencvLAB2standardLAB(boxes)  # shape: (box_num x 3)
    palette_standardLAB = palette.standardLAB[search_index]
    distance = colour.delta_E(boxes_standardLAB[None, :, :], palette_standardLAB[:, None, :],
                              method=deltaE_method)  # shape: (palette_color_num x box_num)
    # relative_L = (palette_standardLAB[:, None, 0] + 5) / (boxes_standardLAB[None, :, 0]+5)
    # relative_L[relative_L < 1] = 1/relative_L[relative_L < 1]
    # distance *= relative_L
    # min_distance_each_color = distance.min(axis=1)   # shape: (palette_color_num)
    min_distance_each_color = np.partition(distance, kth=tolerance_index, axis=1)[:,
                              tolerance_index]  # shape: (palette_color_num)

    # if key_frame:
    #     previous_color_loss_table = palette.DP_loss[palette.nearby_indexes[search_index[:]]]
    # else:
    #     previous_color_loss_table = palette.DP_loss[palette.nearby_indexes[search_index[:]]] \
    #                                 + transloss_beta * palette.nearby_deltaEs[search_index[:]] ** transloss_gamma
    #     # shape: (palette_color_num x nearby_color_num)
    #
    #     # if key_frame:
    #     #     DP_previous_index = np.argmin(palette.DP_loss).repeat(len(search_index))
    #     #     palette.DP_loss[:] = np.inf  # Can be optimized using the previous search_index
    #     #     palette.DP_loss[search_index] = 0
    #     #     previous_color_loss = 0
    #     # else:
    #     #     transloss = transloss_beta * np.exp(-transloss_theta * frame_mean_delta) \
    #     #                 * palette.nearby_deltaEs[search_index[:]] ** transloss_gamma
    #     #     previous_color_loss_table = palette.DP_loss[palette.nearby_indexes[search_index[:]]] + transloss
    #     #     tmp_argmin = np.argmin(previous_color_loss_table, axis=1)  # shape: (palette_color_num)
    #     #     DP_previous_index = palette.nearby_indexes[search_index, tmp_argmin] # shape: (palette_color_num)
    #     previous_color_loss = previous_color_loss_table[range(len(tmp_argmin)), tmp_argmin]

    # Calculate transfer Loss and previous DP index
    transloss = transloss_beta * np.exp(-transloss_theta * frame_mean_delta) \
                * palette.nearby_deltaEs[search_index[:]] ** transloss_gamma
    previous_color_loss_table = palette.DP_loss[palette.nearby_indexes[search_index[:]]] + transloss
    tmp_argmin = np.argmin(previous_color_loss_table, axis=1)  # shape: (palette_color_num)
    DP_previous_index = palette.nearby_indexes[search_index, tmp_argmin]  # shape: (palette_color_num)
    previous_color_loss = previous_color_loss_table[range(len(tmp_argmin)), tmp_argmin]

    # Calcualte distance loss
    distance_loss = min_distance_each_color - distance_standard
    distance_loss[distance_loss > 0] = 0
    distance_loss = np.abs(distance_loss) ** distance_gamma

    DP_loss = previous_color_loss + distance_loss
    #
    # distance_loss = -min_distance_each_color
    # DP_loss = previous_color_loss_table[range(len(tmp_argmin)), tmp_argmin] + distance_loss

    log_statistics["max_min_distance"].append(min_distance_each_color.max())
    log_statistics["max_min_distance_color"].append(palette_standardLAB[np.argmax(min_distance_each_color)])

    return DP_previous_index, DP_loss




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
        delta_E = colour.delta_E(labColor, [L, a, b], method=deltaE_method)
        if delta_E > maxDelta_E:
            maxDelta_E = delta_E
            maxL = L
            maxa = a
            maxb = b
    assert maxL is not None
    assert maxa is not None
    assert maxb is not None
    maxStandardLabColor = np.array([L, a, b])
    print("color with maximum delta E found : {}".format(maxDelta_E))
    print(maxStandardLabColor)
    maxOpencvLabColor = standardLAB2opencvLAB(maxStandardLabColor)[np.newaxis, np.newaxis, :] \
        .astype(np.uint8)
    maxRGBColor = cv.cvtColor(maxOpencvLabColor, cv.COLOR_LAB2RGB).reshape((1, 1, 3))
    plt.imshow(maxRGBColor / 255.0)
    plt.show()


def findMaxDeltaEColorInArray(color_array_a, color_b):
    """
    find the color in color_array_a with max deltaE w.r.t. color_b
    :param color: shape (1) in standard LAB space
    :param color_array: shape (N, 3) in standard LAB space
    :return:
    """
    delta_E = colour.delta_E(color_array_a, color_b, method=deltaE_method)
    index = np.argmax(delta_E)
    max_color = color_array_a[index]
    return index, max_color


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


def showSingleColor(single_color):
    """
    :param bar_array: color bar array ([R, G, B]) with scale (0, 255)
    """
    bar_array = np.array([single_color], dtype=np.int)
    gradient = np.stack((bar_array, bar_array), axis=0)  # convert to 2-dimension
    fig, ax = plt.subplots(nrows=1)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    ax.set_title('colormaps', fontsize=14)
    ax.imshow(gradient, aspect='auto')
    ax.set_axis_off()
    plt.show()
