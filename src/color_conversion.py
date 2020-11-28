import numpy as np


def standardLCH2standardLAB(x):
    """
    :param x: [(0 ~ 100), (-pi/2 ~ pi/2), (-pi/2 ~ pi/2)]
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
    :return: [(0 ~ 100), ..., (-pi/2 ~ pi/2)]
    """
    y = x.astype(np.float32)
    y[..., 0] = x[..., 0]
    y[..., 1] = np.sqrt(x[..., 1]**2 + x[..., 2]**2)
    y[..., 2] = np.arctan(x[...,2] / x[..., 1])
    y[x[..., 1] < 0, 2] += np.pi
    y[np.isnan(y[..., 2]), 2] = np.pi / 2
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
    return y