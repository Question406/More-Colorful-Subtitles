from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
from testHistogram import calcAndDrawHist
# %matplotlib inline 

# pic_file = '../videoSrc/test.png'
pic_file = '../videoSrc/gaomu.jpeg'

img_bgr = cv2.imread(pic_file, cv2.IMREAD_COLOR) #OpenCV读取颜色顺序：BRG 
# img_b = img_bgr[..., 0]
# img_g = img_bgr[..., 1]
# img_r = img_bgr[..., 2]
# fig = plt.gcf()                                  #图片详细信息
# fig = plt.gcf()                                  #分通道显示图片
# fig.set_size_inches(10, 15)

# plt.subplot(221)
# plt.imshow(np.flip(img_bgr, axis=2))             #展平图像数组并显示
# plt.axis('off')
# plt.title('Image')

# plt.subplot(222)
# plt.imshow(img_r, cmap='gray')
# plt.axis('off')
# plt.title('R')

# plt.subplot(223)
# plt.imshow(img_g, cmap='gray')
# plt.axis('off')
# plt.title('G')

# plt.subplot(224)
# plt.imshow(img_b, cmap='gray')
# plt.axis('off')
# plt.title('B')
# plt.show()


# img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
# img_ls = img_lab[..., 0]
# img_as = img_lab[..., 1]
# img_bs = img_lab[..., 2] 

# # 分通道显示图片
# fig = plt.gcf()
# fig.set_size_inches(10, 15)

# plt.subplot(221)
# plt.imshow(img_lab)
# plt.axis('off')
# plt.title('L*a*b*')

# plt.subplot(222)
# plt.imshow(img_ls, cmap='gray')
# plt.axis('off')
# plt.title('L*')

# plt.subplot(223)
# plt.imshow(img_as, cmap='gray')
# plt.axis('off')
# plt.title('a*')

# plt.subplot(224)
# plt.imshow(img_bs, cmap='gray')
# plt.axis('off')
# plt.title('b*')

# plt.show()  


# img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# img_gray_hist = cv2.calcHist([img_b], [0], None, [256], [0, 256])
# plt.plot(img_gray_hist)
# plt.title('img_b Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of Pixels')
# plt.show()
color = ('r', 'g', 'b')
plt.subplot(221)
for i, col in enumerate(color):
    histr = cv2.calcHist([img_bgr], [i], None, [256 / 8], [0, 255])
    print(histr.shape)
    print(len(histr))
    plt.plot(histr, color = col)
    plt.xlim([0, 256 / 8 - 1])

plt.subplot(222)
for i, col in enumerate(color):
    histr = cv2.calcHist([img_bgr], [i], None, [256], [0, 255])
    print(histr.shape)
    print(len(histr))
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
plt.show()

# import cv2 as cv
# import numpy as np

# img = cv2.imread(pic_file)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # 灰度图均衡化
# equ = cv2.equalizeHist(gray)
# # 水平拼接原图和均衡图
# result1 = np.hstack((gray, equ))
# cv2.imwrite('grey_equ.png', result1)
# # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
# (b, g, r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # 合并每一个通道
# equ2 = cv2.merge((bH, gH, rH))
# # 水平拼接原图和均衡图
# result2 = np.hstack((img, equ2))
# cv2.imwrite('bgr_equ.png', result2)

# img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# (l, a, b) = cv2.split(img_lab)
# lH = cv2.equalizeHist(l)
# aH = cv2.equalizeHist(a)
# bH = cv2.equalizeHist(b)

# equ3 = cv2.merge((lH, aH, bH))
# result3 = np.hstack((img_lab, equ3))
# result4 = cv2.cvtColor(result3, cv2.COLOR_LAB2BGR)
# cv2.imwrite('lab_equ.png', result4)