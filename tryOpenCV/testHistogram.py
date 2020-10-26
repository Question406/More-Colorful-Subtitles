import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./videoSrc/test.png')
b, g, r = cv2.split(img)
print(img.shape)

plt.show()
plt.figure()
hists,bins = np.histogram(b.flatten(),256,[0,256])  #和注释掉的绘图效果一样，不过是曲线
plt.plot(hists,color='r')
plt.show()