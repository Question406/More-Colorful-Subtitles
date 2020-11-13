# from cv2 import cv2 as cv
# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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


def _d_textRegion2textColor(bbox, textColor):
    """
    :param bbox: text region of a single character, LAB color space
    :param textColor: color we try to put on text, LAB color space
    :return: euclidean distance between color in the bounding box and text color,
    """
    assert len(bbox.shape) == 3
    colorBBox = bbox.reshape((-1, 3))
    return np.mean(np.sqrt(np.sum((colorBBox - textColor) ** 2, axis=-1)))


def d_textRegion2textColor(bboxs, textColor):
    """
    :param bboxs: text region of all characters, LAB color space
    :param textColor: color we try to put on text, LAB color space
    :return: minimum textRegion2textColor distance:
             min(d(bbox, textColor))
    """
    d = [_d_textRegion2textColor(bbox, textColor) for bbox in bboxs]
    return np.min(d)


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


def calculateLoss(image, frame, im, lastStatus, colors, text, anchor, font=getFont('Consolas', 32)):
    epsilon = 5
    # bounding box of entire text
    #    bbox = image.textbbox(anchor, text, font)
    # left upper bound of bounding box
    lastl, u = anchor[0], anchor[1]

    chboxs = []
    # for each character, find its bounding box
    for ch in text:
        box = image.textbbox((lastl, u), ch, font=font)

        image.rectangle(box, fill=None, outline=(255, 0, 0))

        chboxs.append(box)
        lastl = box[2]
    boxs = [frame[chbox[1]: chbox[3], chbox[0]:chbox[2]] for chbox in chboxs]
    # print(boxs[0])

    # temp = PILToCV2(image)
    # temp = cv.cvtColor(np.asarray(im), cv.COLOR_RGB2BGR)
    # for chbox in chboxs:
    # frame[chbox[0]: chbox[2], chbox[1]:chbox[3]] = (255, 0, 0)
    #    frame[chbox[1]: chbox[3], chbox[0]:chbox[2]] = (255, 0, 0)

    # cv.imwrite('temp.png', frame)

    # [(loss, from which color)...]
    resStatus = np.zeros((len(colors), 2))

    for (i, status) in enumerate(resStatus):
        curColorLoss = d_textRegion2textColor(boxs, colors[i])
        # colorLoss = 1e30
        status[1] = max(i - epsilon, 0) + np.argmin(lastStatus[max(i - epsilon, 0): min(len(colors), i + epsilon), 0])
        # for j in range(max(i - epsilon, 0), min(len(colors), i + epsilon)):
        # if (lastStatus[j][0] + transColorLoss(colors[j], colors[i]) < colorLoss):
        #     colorLoss = lastStatus[j][0] + transColorLoss(colors[j], colors[i])
        #     # from which color
        #     status[1] = j
        # if (lastStatus[j][])
        colorLoss = lastStatus[int(status[1])][0]
        colorLoss += curColorLoss
        # color loss
        status[0] = colorLoss
    return resStatus
