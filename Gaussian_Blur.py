
#  https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
#  https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
#  https://docs.opencv.org/2.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html



import cv2
import numpy as np

def unsharp_mask(img, blur_size = (9,9), imgWeight = 1.0, gaussianWeight = -0.5):
    gaussian = cv2.GaussianBlur(img, (5,5), 0)
    return cv2.addWeighted(img, imgWeight, gaussian, gaussianWeight, 0)

img_file = 'lemonade.png'
img = cv2.imread(img_file, cv2.IMREAD_COLOR)
img = cv2.blur(img, (5, 5))
img = unsharp_mask(img)
img = unsharp_mask(img)
img = unsharp_mask(img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

thresh = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
_, contours, heirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(contours, key = cv2.contourArea, reverse = True)
#for cnt in cnts:
canvas_for_contours = thresh.copy()
cv2.drawContours(thresh, cnts[:-1], 0, (0,255,0), 3)
cv2.drawContours(canvas_for_contours, contours, 0, (0,255,0), 3)
cv2.imshow('Result', canvas_for_contours - thresh)
cv2.imwrite("Gaussian_Blur_Result.png", canvas_for_contours - thresh)
cv2.waitKey(0)
