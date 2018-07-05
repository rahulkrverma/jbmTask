
import os
import numpy as np
import cv2


def preProcess(imgArr):

	img_gray = cv2.bilateralFilter(imgArr, 15, 19, 19)
	# img_gray = cv2.GaussianBlur(imgArr, (3, 3), 0)
	# img_edge = cv2.Canny(img_gray, 10, 50)

	img_edge = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize = 5)
	img_thrus = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 3)

	return img_thrus