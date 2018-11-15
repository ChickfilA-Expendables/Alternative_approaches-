
# https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
# http://opencvexamples.blogspot.com/2013/10/sobel-edge-detection.html
#  https://shahsparx.me/edge-detection-opencv-python-video-image/


import cv2
import numpy as np
from matplotlib import pyplot as plt

# loading image
#img0 = cv2.imread('SanFrancisco.jpg',)
img0 = cv2.imread('lemonade.png',)

# converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(gray,(3,3),0)

# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_16U)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y cv.BORDER_ISOLATED

# watershed the images
#markers = cv.watershed(img,markers)
#img[markers == -1] = [255,0,0]

plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])  # original
plt.subplot(2,3,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([]) # laplacian
plt.subplot(2,3,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([]) # sobel x
plt.subplot(2,3,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([]) # sobel y

# sobel x + sobel y
combined = cv2.add(sobelx,sobely)
plt.subplot(2,3,5),plt.imshow(combined,cmap = 'gray')
plt.title('combined'), plt.xticks([]), plt.yticks([])

# combined with original sobel x+ y + original
"""
supercombined = cv2.add(combined, img)
plt.subplot(2,3,6),plt.imshow(supercombined,cmap = 'gray')
plt.title('super_combined'), plt.xticks([]), plt.yticks([])
"""
plt.show()
