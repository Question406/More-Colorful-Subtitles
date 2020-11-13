import os

from PIL import ImageDraw
from cv2 import cv2 as cv

from utils import (drawImageSingleText, CV2ToPIL, PILToCV2, getTextInfoPIL, funcTime)


# imagePath = "./videoSrc/test.png"
# outputPath = "./temp.png"
# outputF = open(outputPath, "wb")
#
# fontpath = "fonts/Dyuthi-Regular.ttf"
# font = ImageFont.truetype(fontpath, 32)
# color = (0, 255, 0)
# # with Image.open(imagePath).convert("RGBA") as im:
# with Image.open(imagePath) as im:
#     draw = ImageDraw.Draw(im)
#     drawImageSingleText(draw, 'Hello World!', font='Consolas', anchor=(0, 0))
#     im.save(outputF, "PNG")
# outputF.close()

def _main(*args):
    # video src
    srcName = args[0]
    src = srcName.strip(".mp4").strip(".flv")
    videoSrc = "./videoSrc/%s" % srcName
    outputDir = "./videoOutput/%s" % src
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    result_video = "%s/%s-Subtitle.mp4" % (outputDir, src)
    cap = cv.VideoCapture(videoSrc)
    print("read done")
    # video FPS
    fps_video = cap.get(cv.CAP_PROP_FPS)
    # video format
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # frame width and height
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    videoWriter = cv.VideoWriter(result_video, fourcc, fps_video, (frame_width, frame_height))
    frame_id = 0
    print("transfering %s" % srcName)
    print(frame_width, " ", frame_height)
    output = []
    k = 5
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_id += 1
            if frame_id % 100 == 0:
                print("hello", frame_id // 100)
            # text coordinate
            text = 'frame_%s' % frame_id
            imgH, imgW = frame_height, frame_width
            fontScale, fontThickness = 1.5, 2
            im = CV2ToPIL(frame)
            draw = ImageDraw.Draw(im)
            textWidth, textHeight = getTextInfoPIL(draw, text)
            drawImageSingleText(draw, text, font='Consolas', anchor=(imgW // 2 - textWidth // 2, imgH - k * textHeight))
            frame = PILToCV2(im)
            videoWriter.write(frame)

        else:
            videoWriter.release()
            break

    return 0


funcTime(_main, 'demo_Trim.mp4')
