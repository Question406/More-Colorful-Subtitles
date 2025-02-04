import numpy as np
from cv2 import cv2 as cv

def standardLAB2visibleLAB(x):
    """
    :param x: LAB color in standard form [(0 ~ 100), (-128 ~ 127), (-128 ~ 127)]
    :return: visible LAB color in standard form [(0 ~ 100), (-128 ~ 127), (-128 ~ 127)]
    """
    y = standardLAB2RGB(x)
    tmp_2d_shape = (1, -1, 3)
    y = cv.cvtColor(y.reshape(tmp_2d_shape), cv.COLOR_RGB2LAB).reshape(y.shape)
    y = opencvLAB2standardLAB(y)
    return y

def standardLAB2RGB(x):
    y = standardLAB2opencvLAB(x)
    tmp_2d_shape = (1, -1, 3)
    y = cv.cvtColor(y.reshape(tmp_2d_shape), cv.COLOR_LAB2RGB).reshape(y.shape)
    return y

def standardLCH2standardLAB(x):
    """
    :param x: [(0 ~ 100), ..., (-pi ~ pi)]
    :return: [(0 ~ 100), (-128 ~ 127), (-128 ~ 127)]
    """
    y = x.astype(np.float32)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1] * np.cos(x[..., 2])
    y[..., 2] = x[..., 1] * np.sin(x[..., 2])
    return y


def standardLAB2standardLCH(x):
    """
    :param x: [(0 ~ 100), (-128 ~ 127), (-128 ~ 127)]
    :return: [(0 ~ 100), ..., (-pi ~ pi)]
    """
    y = x.astype(np.float32)
    y[..., 0] = x[..., 0]
    y[..., 1] = np.sqrt(x[..., 1]**2 + x[..., 2]**2)
    y[..., 2] = np.arctan(x[..., 2] / x[..., 1])
    y[np.logical_and(x[..., 1] <= 0, x[..., 2] > 0), 2] += np.pi
    y[np.logical_and(x[..., 1] < 0, x[..., 2] < 0), 2] -= np.pi
    y[np.isnan(y[..., 2]), 2] = np.pi / 2
    return y

def oppositeStandardLCH(x):
    y = x.astype(np.float32)
    y[x[..., 2] <= 0, 2] += np.pi
    y[x[..., 2] > 0, 2] -= np.pi
    return y


def opencvLAB2standardLAB(x):
    """
    :param x: [(0 ~ 255), (0 ~ 255), (0 ~ 255)]
    :return:  [(0 ~ 100), (-128 ~ 127), (-128 ~ 127)]
    """
    y = x.astype(np.float32)
    y[..., 0] = x[..., 0] / 255.0 * 100
    y[..., 1] = x[..., 1] - 128.0
    y[..., 2] = x[..., 2] - 128.0
    return y


def standardLAB2opencvLAB(x):
    """
    :param x: [(0 ~ 100), (-128 ~ 127), (-128 ~ 127)]
    :return: [(0 ~ 255), (0 ~ 255), (0 ~ 255)]
    """
    y = x.astype(np.float32)
    y[..., 0] = x[..., 0] / 100.0 * 255
    y[..., 1] = x[..., 1] + 128.0
    y[..., 2] = x[..., 2] + 128.0
    return y.astype(np.uint8)