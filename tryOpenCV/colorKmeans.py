# import the necessary packages
import random

import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
from sklearn.cluster import KMeans


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

def plot_colors(hist, centroids):
    """
    :param hist: percentage of clusters
    :param centroids: cluster color RGB
    :return: a bar that visualize the percentage of cluster colors
    """
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX
    # return the bar chart
    return bar

def _getClusters(image, k = 3):
    """
    :param image: source color matrix (reshaped to 2d)
    :param k: the cluster number
    :return: cluster color(in the original color space) and percentage
    """
    clt = KMeans(n_clusters=k)
    clt.fit(image)
    hist = centroid_histogram(clt)
    # bar = plot_colors(hist, clt.cluster_centers_)
    return sorted(zip(hist, clt.cluster_centers_), reverse=True)

def getClusters(image, k = 3):
    """
    :param image: source image matrix
    :param k: the cluster number
    :return: cluster color and its percentage(sort by percentage in descending order)
    """
    temp = image.reshape((image.shape[0] * image.shape[1], 3))
    return _getClusters(temp, k)
    
def showImage(src, name=""):
    """
    :param src: source color matrix
    :param name: figure name
    """
    # cv2 use BGR color, so we need to transfer it to RGB
    tSrc = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    plt.figure(name)
    plt.axis("off")
    plt.imshow(tSrc)

def visualizeColorClusters(image, k = 3):
    """
    :param image: source color matrix
    :param k: the cluster number
    """
    temp = image.reshape((image.shape[0] * image.shape[1], 3))
    temp = _getClusters(temp, k)
    hist, color = zip(*temp)
    bar = plot_colors(hist, color)
    showImage(image, "image")
    showImage(bar, "bar")
    plt.show()


# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image_path = '../videoSrc/test.png'
image = cv2.imread(image_path)
# BGR-->RGB cv to matplotlib show
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # show our image
# plt.figure()
# plt.axis("off")
# plt.imshow(image)

# # reshape the image to be a list of pixels
image = np.array(random.sample(list(image.reshape((-1, 3))), 500))

# image = image.reshape((-1, 3))
# # cluster the pixel intensities
clt = KMeans(n_clusters=5)
clt.fit(image)
# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)
# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
