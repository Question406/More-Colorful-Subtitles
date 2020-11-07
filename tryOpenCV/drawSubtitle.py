from cv2 import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# imgH, imgW, _ = img_bgr.shape
# print(imgW, " ", imgH)

# text = 'This is a test text %s' %str(12345)
# res, baseLine = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 2, 2)
# width, height = res[0], res[1]
# print(width, height)

# cv.putText(img_bgr, text, (imgW // 2 - width // 2, imgH - height), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
# cv.imwrite('test_textImg.png', img_bgr)

def addSubTitle(img, text, fontColor, fontScale=1.5, fontThickness=1):
    """
    :param img: RGB img matrix
    :param text: the text to add
    :param fontColor: fontCOlor
    :return: a img matrix after adding subtitle
    """

    imgH, imgW, _ = img.shape
    res, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, fontScale, fontThickness)
    width, height = res
    cv.putText(img, text, (imgW // 2 - width // 2, imgH -  5 * height), cv.FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontThickness)


pic_file = '../videoSrc/bar_code.jpeg'
img_bgr = cv.imread(pic_file, cv.IMREAD_COLOR)
text = "this is a test text"

fontScale, fontThickness = 1.5, 2

imgH, imgW, _ = img_bgr.shape
res, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, fontScale, fontThickness)
width, height = res

LLCorner = (imgW // 2 - width // 2, imgH - height)
# get boundingbox
boundingBox = img_bgr[LLCorner[1]- 5 * height : LLCorner[1], LLCorner[0] : LLCorner[0] + width, :]
boundingLAB = cv.cvtColor(boundingBox, cv.COLOR_BGR2LAB)

(l, a, b) = cv.split(boundingLAB)
print(np.max(l))
print(np.min(l))
# get RGB color of subtitle
average_l = np.mean(l) 
average_a = np.mean(a)
average_b = np.mean(b)

print("original LAB: ", average_l, " ", average_a - 128, " ", average_b - 128)
# average_l = 256 - average_l
# average_l = 
# average_a = 256 - average_a
# average_b = 256 - average_b
# average_l = 100 - np.mean(l)
average_a = 121 + 128
average_b = 123 + 128
print("subtitle LAB: ", average_l, " ", average_a - 128, " ", average_b - 128)
temp = boundingLAB
temp[0][0][0] = average_l
temp[0][0][1] = average_a
temp[0][0][2] = average_b

res = (average_l, average_a, average_b)
resRGB = cv.cvtColor(temp, cv.COLOR_LAB2BGR)[0][0]
resRGB = [int(x) for x in resRGB]
print(resRGB[2], " ", resRGB[1], " ", resRGB[0])

# add subtitle
addSubTitle(img_bgr, text, resRGB, fontScale, fontThickness)
cv.imwrite('test_textImg.png', img_bgr)