#!/usr/bin/env python
# -*- coding:utf-8 -*-
# import sys
from cv2 import cv2
from drawSubtitle import addSubTitle
import numpy as np

# sys.path.append("./videoSrc/")
# video = "../videoSrc/TWICE-What_Is_Love.mp4"
# srcName = "BLACKPINK-Kill_This_Love"
srcName = "TWICE-What_Is_Love"
# srcName = "WesternSichuan"
video = "../videoOutput/%s.mp4"%srcName
result_video = "%s-Subtitle.mp4"%srcName
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

k = 5

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

        LLCorner = (imgW // 2 - width // 2, imgH - k * height)
        # get boundingbox
        boundingBox = frame[LLCorner[1]- k * height : LLCorner[1], LLCorner[0] : LLCorner[0] + width, :]
        boundingHSV = cv2.cvtColor(boundingBox, cv2.COLOR_BGR2HSV)
        (h, s, v) = cv2.split(boundingHSV)

        # get RGB color of subtitle
        # average_h = int(180. - np.mean(h))
        average_h = (int(np.mean(h)) + 90) % 180
        average_s = int(np.mean(s))
        average_v = int(255 - np.mean(v))
        
        temp = boundingHSV
        temp[0][0][0] = average_h
        temp[0][0][1] = average_s
        temp[0][0][2] = average_v

        res = (average_h, average_s, average_v)
        resRGB = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)[0][0]
        resRGB = [int(x) for x in resRGB]
        # temp = [str(resRGB[2]), str(resRGB[1]), str(resRGB[0])]
        temp = [str(int(np.mean(h))), str(int(np.mean(s))), str(int(np.mean(v))) , " : ", str(int(average_h)), str(int(average_s)), str(int(average_v))]
        output.append(" ".join(temp))

        addSubTitle(frame, text, resRGB, fontScale, fontThickness)
        videoWriter.write(frame)
    else:
        videoWriter.release()
        break
with open("%s-hsv.txt"%srcName, "w") as f:
    f.write("\n".join(output))