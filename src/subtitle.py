import pysrt

from utils import (getFont)


class subtitle:
    def init(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text
        return self

    def __init__(self, sub, videoFPS):
        if not sub is None:
            self.start = int(sub.start.ordinal / 1000 * videoFPS)
            self.end = int(sub.end.ordinal / 1000 * videoFPS)
            self.text = sub.text
        else:
            self.start = -1
            self.end = -1
            self.text = 'Error'

    def __str__(self):
        return "[{} : {} : {}]".format(self.start, self.end, self.text)

def drawSubtitle(image, text, anchor, font=getFont('Consolas', 32), color=(255, 0, 0)):
    # color = (int(color[0]) for c in color]
    # print(color)
    color = (int(color[0]), int(color[1]), int(color[2]))
    image.text(anchor, text, font=font, fill=color)
    return image


def processSRT(srtPath, videoFPS):
    """
    :param srtName: file name of the srt file
    :param videoFPS: the videoFPS
    :return: a list of all subs, containing start frame_id and end frame id and its context
    """

    subs = pysrt.open(srtPath)
    res = [subtitle(sub, videoFPS) for sub in subs]
    return res
