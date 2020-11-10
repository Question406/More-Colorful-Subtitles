#!/usr/bin/env python
# -*- coding:utf-8 -*-
# import sys
from cv2 import cv2
from drawSubtitle import addSubTitle
import numpy as np

# sys.path.append("./videoSrc/")
video = "../videoSrc/TWICE-What_Is_Love.mp4"
result_video = "TWICE-What_Is_Love-Subtitle.mp4"
#读取视频
cap = cv2.VideoCapture(video)
print("read done")
#获取视频帧率
fps_video = cap.get(cv2.CAP_PROP_FPS)
#设置写入视频的编码格式
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#获取视频宽度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#获取视频高度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoWriter = cv2.VideoWriter(result_video, fourcc, fps_video, (frame_width, frame_height))

print(fourcc)
print(frame_width)
print(frame_height)

frame_id = 0
print("transfer")
output = []

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame_id += 1
        if frame_id % 100 == 0:
            print("hello")
        left_x_up = int(frame_width / frame_id)
        left_y_up = int(frame_height / frame_id)
        right_x_down = int(left_x_up + frame_width / 10)
        right_y_down = int(left_y_up + frame_height / 10)
        #文字坐标
        text = 'frame_%s '%frame_id
        fontScale, fontThickness = 1.5, 2
        imgH, imgW, _ = frame.shape
        res, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontThickness)
        width, height = res

        LLCorner = (imgW // 2 - width // 2, imgH - 5 * height)
        # get boundingbox
        boundingBox = frame[LLCorner[1]- 5 * height : LLCorner[1], LLCorner[0] : LLCorner[0] + width, :]
        boundingLAB = cv2.cvtColor(boundingBox, cv2.COLOR_BGR2LAB)
        (l, a, b) = cv2.split(boundingLAB)
        # frame_LAB = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        # (l, a, b) = cv2.split(frame_LAB)
        # print(np.max(l))
        # print(np.min(l))
        # get RGB color of subtitle
        average_l = (256 - np.mean(l))
        average_a = np.mean(a)
        average_b = np.mean(b)
        average_a = 256 - average_a
        average_b = 256 - average_b
        temp = boundingLAB
        temp[0][0][0] = average_l
        temp[0][0][1] = average_a
        temp[0][0][2] = average_b

        res = (average_l, average_a, average_b)
        resRGB = cv2.cvtColor(temp, cv2.COLOR_LAB2BGR)[0][0]
        resRGB = [int(x) for x in resRGB]
        # temp = [str(resRGB[2]), str(resRGB[1]), str(resRGB[0])]
        temp = [ str(int(average_l)), str(int(average_a)), str(int(average_b))]
        output.append(" ".join(temp))

        # cv.putText(frame, 'frame_%s' %frame_id, (word_x, word_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (55,255,155), 2)
        addSubTitle(frame, text, resRGB, fontScale, fontThickness)
        videoWriter.write(frame)
    else:
        videoWriter.release()
        break
with open("rgb.txt", "w") as f:
    f.write("\n".join(output))