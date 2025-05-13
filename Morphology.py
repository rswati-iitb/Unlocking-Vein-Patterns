# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 18:16:35 2022

@author: 91887
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('CLAHE Image.png',0)
kernel = np.ones((5,5),np.uint8)
#erosion = cv.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img,kernel,iterations = 1)
#opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing)
plt.xticks([]), plt.yticks([])
cv2.imwrite("morphology.png", closing)
