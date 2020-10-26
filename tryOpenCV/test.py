#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
video = "demo.mp4"
result_video = "demo-result.mp4"
#读取视频
cap = cv2.VideoCapture(video)
#获取视频帧率
fps_video = cap.get(cv2.CAP_PROP_FPS)
#设置写入视频的编码格式
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#获取视频宽度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#获取视频高度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoWriter = cv2.VideoWriter(result_video, fourcc, fps_video, (frame_width, frame_height))
frame_id = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame_id += 1
        left_x_up = int(frame_width / frame_id)
        left_y_up = int(frame_height / frame_id)
        right_x_down = int(left_x_up + frame_width / 10)
        right_y_down = int(left_y_up + frame_height / 10)
        #文字坐标
        word_x = left_x_up + 5
        word_y = left_y_up + 25
#        cv2.rectangle(frame, (left_x_up, left_y_up), (right_x_down, right_y_down), (55,255,155), 5)
        cv2.putText(frame, 'frame_%s' %frame_id, (word_x, word_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (55,255,155), 2)
        videoWriter.write(frame)
    else:
        videoWriter.release()
        break
