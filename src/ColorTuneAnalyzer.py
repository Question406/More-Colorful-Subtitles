import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from cv2 import cv2 as cv
from color_conversion import *

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class ColorTuneAnalyzer:
    def __init__(self, frame_width, frame_height, sample_num=100, n_cluser=5):
        self.sampleNum = sample_num
        self.n_cluster = n_cluser
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sample_x_indices = np.random.randint(0, frame_width, size=sample_num)
        self.sample_y_indices = np.random.randint(0, frame_height, size=sample_num)

    @ignore_warnings(category=ConvergenceWarning)
    def analyzeImage(self, img):
        """
        :param img:
        :return centroids: the cluster point in standard LAB space after ordering
        """
        clt = KMeans(n_clusters=self.n_cluster)
        # Choose which color space to cluster is essential, here we choose LAB space
        sample_color = cv.cvtColor(img[None, self.sample_y_indices, self.sample_x_indices, :],
                                   cv.COLOR_BGR2LAB).squeeze()
        clt.fit(sample_color)
        centroids = clt.cluster_centers_
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (centroids_percent, _) = np.histogram(clt.labels_, bins=numLabels)
        order_index = np.argsort(centroids_percent)[::-1]   # from main tune to minor tune
        return opencvLAB2standardLAB(centroids[order_index])


def analysisVideoColorTune(cap, sample_num=500, n_cluster=5):
    plotHeight = 50

    def centroid_histogram(clt):
        """
        :param clt: the cluster fit by k-means
        :return: the percentage histogram of clusters
        """
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram
        return hist

    def plotColorTune(sample_palattes):
        sample_palattes = np.array(sample_palattes).astype(np.int).swapaxes(0, 1)
        fig, ax = plt.subplots(nrows=1)
        fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
        ax.set_title('colormaps', fontsize=14)
        ax.imshow(sample_palattes, aspect='auto')
        ax.set_axis_off()
        plt.show()

    print("Video Color Tune Analysis Begin!")
    clt = KMeans(n_clusters=n_cluster)
    cap.set(1, 1)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    sample_x_indices = np.random.randint(0, frame_width, size=(sample_num))
    sample_y_indices = np.random.randint(0, frame_height, size=(sample_num))

    current_frame = 0
    start = time.time()
    sample_palattes = []
    while cap.isOpened():
        current_frame += 1
        ret, frame = cap.read()
        if ret and current_frame < 500:
            if current_frame % 100 == 0:
                print("Frame:{}, Time:{}".format(current_frame, time.time() - start))

            # There we temporarily choose RGB space to cluster for convenience
            sample_color = cv.cvtColor(frame[None, sample_y_indices, sample_x_indices, :], cv.COLOR_BGR2RGB).squeeze()
            clt.fit(sample_color)
            centroids_percent = centroid_histogram(clt)
            centroids = clt.cluster_centers_

            # Collect Palatte Information
            index = np.argsort(centroids_percent)
            current_palatte = np.zeros(shape=(plotHeight, 3))
            current_height = 0
            for i in range(len(clt.cluster_centers_)):
                next_height = int(current_height + plotHeight * centroids_percent[index[i]])
                current_palatte[current_height: next_height] = centroids[[index[i]]]
                current_height = next_height
            sample_palattes.append(current_palatte)
        else:
            plotColorTune(sample_palattes)
            break
    print("Analysis Finished")
