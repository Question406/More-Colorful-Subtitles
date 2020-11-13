from utils import (getFont)


def drawSubtitle(image, text, anchor, font=getFont('Consolas', 32), color=(255, 0, 0)):
    # color = (int(color[0]) for c in color]
    # print(color)
    color = (int(color[0]), int(color[1]), int(color[2]))
    image.text(anchor, text, font=font, fill=color)
    return image


def getColors(image, text, anchor, font='Consolas', fontsize=32):
    return
