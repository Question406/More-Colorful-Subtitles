import re
import time

from utils import (getFont)


def drawSubtitle(image, text, anchor, font=getFont('Consolas', 32), color=(255, 0, 0)):
    # color = (int(color[0]) for c in color]
    # print(color)
    color = (int(color[0]), int(color[1]), int(color[2]))
    image.text(anchor, text, font=font, fill=color)
    return image


def processSRT(srtName, videoFPS):
    TIME_PATTERN = '%02d:%02d:%02d,%03d'
    TIME_REPR = 'SubRipTime(%d, %d, %d, %d)'
    RE_TIME_SEP = re.compile(r'\:|\.|\,')
    RE_INTEGER = re.compile(r'^(\d+)')
    SECONDS_RATIO = 1000
    MINUTES_RATIO = SECONDS_RATIO * 60
    HOURS_RATIO = MINUTES_RATIO * 60

    srtPath = './subtitles/srt/%s.srt' % srtName

    tempVal = ((0, 0), "None")
    resList = []
    with open(srtPath, 'r') as f:
        for ind, val in enumerate(f.readlines()):
            if ind % 4 == 1:
                time.strftime("%H:%m:S")
            elif ind % 4 == 2:

                return

    return
