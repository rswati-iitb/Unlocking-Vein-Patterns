import cv2
import numpy as np
from matplotlib import pyplot as plt
# loading image
img = cv2.imread('CLAHE Image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting to gray scale
laplacian = cv2.Laplacian(img,cv2.CV_64F)  # convolute with proper kernels
cv2.imshow("Laplacian Image.png",laplacian )
plt.figure(figsize=(11,6))
plt.subplot(131), plt.imshow(img, cmap='gray'),plt.title('CLAHE Image')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img, cmap='gray'),plt.title('Laplacian')
plt.xticks([]), plt.yticks([])
#plt.subplot(133), plt.imshow(img, cmap='gray'),plt.title('Resulting image')
#plt.xticks([]), plt.yticks([])
plt.show()

