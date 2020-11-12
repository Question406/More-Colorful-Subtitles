from cv2 import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from colorKmeans import *
import json

def addSubTitle(img, text, fontColor, fontScale=1.5, fontThickness=1):
    """
    :param img: BGR img matrix
    :param text: the text to add
    :param fontColor: fontColor in BGR color space
    :return: img matrix after adding subtitle
    """

    imgH, imgW, _ = img.shape
    res, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, fontScale, fontThickness)
    width, height = res
    cv.putText(img, text, (imgW // 2 - width // 2, imgH -  5 * height), cv.FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontThickness)

# def getColor(boundingLAB):
#     """
#     :param boundingLAB: the bounding box of LAB color space
#     """
#     colorClusters = getClusters(boundingLAB, k=3)
#     color, percentage = zip(*colorClusters)
    
#     if percentage[0] > 0.7:
#         resColor = np.mean(color,axis=1)
#         print(resColor)
#     elif (percentage[0] - percentage[1] < 0.1 and np.linalg.norm(np.mean(color, axis=1), axis=1, keepdims=True) < ) :
        

# pic_file = '../videoSrc/test.png'
# img_bgr = cv.imread(pic_file, cv.IMREAD_COLOR)
# # img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
# img_LAB = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
# text = "this is a test text"

# fontScale, fontThickness = 1.5, 2

# imgH, imgW, _ = img_bgr.shape
# res, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, fontScale, fontThickness)
# width, height = res

# LLCorner = (imgW // 2 - width // 2, imgH - height)
# # get boundingbox
# boundingBox = img_bgr[LLCorner[1]- 5 * height : LLCorner[1], LLCorner[0] : LLCorner[0] + width, :]
# boundingLAB = cv.cvtColor(boundingBox, cv.COLOR_BGR2LAB)

# # (l, a, b) = cv.split(boundingLAB)
# (l, a, b) = cv.split(img_LAB)
# # overallClusters = getClusters(img_bgr)
# # subtitleClusters = getClusters(boundingBox)

# # # get RGB color of subtitle
# average_l = np.mean(l) 
# average_a = np.mean(a)
# average_b = np.mean(b)

# print("original LAB: ", average_l, " ", average_a - 128, " ", average_b - 128)
# average_a = 256 - average_a
# average_b = 256 - average_b
# print("subtitle LAB: ", average_l, " ", average_a - 128, " ", average_b - 128)

# temp = np.array([average_l, average_a, average_b], dtype=boundingLAB[0][0].dtype).reshape(1, 1, 3)

# resRGB = cv.cvtColor(temp, cv.COLOR_LAB2BGR)[0][0]
# resRGB = [int(x) for x in resRGB]
# # print(resRGB[2], " ", resRGB[1], " ", resRGB[0])

# # # add subtitle
# addSubTitle(img_bgr, text, resRGB, fontScale, fontThickness)
# cv.imwrite('test_textImg.png', img_bgr)