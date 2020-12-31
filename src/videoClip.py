import os
from cv2 import cv2 as cv

srcName = "TWICE-What_Is_Love_000.mp4"
start_frame = 2058
end_frame = 2120
videoSrc = './videoOutput/TWICE-What_Is_Love/%s' % srcName

def videoClip(cap, start_frame, end_farme):
    # output path
    resultPath = "output.mp4"
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv.VideoWriter(resultPath, fourcc, cap.get(cv.CAP_PROP_FPS), (frame_width, frame_height))

    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if ret:
            print('123')
            videoWriter.write(frame)
        else:
            videoWriter.release()
            break

cap = cv.VideoCapture(videoSrc)
videoClip(cap, start_frame, end_frame)

