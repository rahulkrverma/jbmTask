

import os
import numpy as np
import cv2
import glob


def padding(imgArr):
	rows, cols = imgArr.shape
	bottom= imgArr[rows - 2 : rows, 0 : cols]
	mean= cv2.mean(bottom)[0]
	bordersize = 20
	imgPadd = cv2.copyMakeBorder(imgArr, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[mean, mean, mean])
	return imgPadd

def rotateImg(imgArr, rtFactor):
	rows, cols = imgArr.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2), rtFactor, 1)
	imgRt = cv2.warpAffine(imgArr, M, (cols, rows))
	return imgRt

def imgErrode(imgArr):
	kernel = np.ones((1, 2), np.uint8)
	imgEros = cv2.erode(imgArr, kernel, iterations = 1)
	return imgEros
	

def imgDilation(imgArr):
	kernel = np.ones((2, 2), np.uint8)
	dilation = cv2.dilate(imgArr, kernel, iterations = 1)
	return dilation



if __name__ == '__main__':
	
	dataPath = '../All_61326/train_61326/'
	augPath = '/home/user/Desktop/jbmTask/imgAug_edge'

	rt_mat = [5, -1, 10, -10]

	classList = os.listdir(dataPath)
	for className in classList:
		if not className.endswith('.DS_Store'):
			classPath = os.path.join(dataPath, className)

			classAug = os.path.join(augPath, className)
			if not os.path.exists(classAug):
				os.mkdir(classAug)

			imgList = os.listdir(classPath)
			for img in imgList:
				if not img.endswith('.DS_Store'):
					imgPath = os.path.join(classPath, img)

					imgArr = cv2.imread(imgPath, 0)
					# imgArr = cv2.resize(imgArr, (720, 480), interpolation=cv2.INTER_CUBIC)

					# blur = cv2.medianBlur(imgArr, 5)
					# blur = cv2.GaussianBlur(imgArr,(5,5),0)

					# th3 = cv2.adaptiveThreshold(imgArr_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
					# edges = cv2.Canny(imgArr, 50, 50)
					
					# kernel = np.ones((5,5),np.float32)/17
					# dst = cv2.filter2D(imgArr,-1,kernel)

					# blur = cv2.blur(imgArr,(5,5))
					# edges = cv2.Canny(blur, 50, 70)

					img_gray = cv2.bilateralFilter(imgArr, 15, 19, 19)
					# img_gray = cv2.GaussianBlur(imgArr, (3, 3), 0)
					# img_edge = cv2.Canny(img_gray, 10, 50)

					# img_edge = cv2.Laplacian(imgArr ,cv2.CV_64F)

					img_edge = cv2.Sobel(img_gray,cv2.CV_64F, 0, 1, ksize=5)
					
					# ret, th1 = cv2.threshold(img_edge, 100, 255,cv2.THRESH_BINARY)

					th3 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 3)

					# _, cnts, _ = cv2.findContours(img_edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
					# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
					# outline = np.zeros(imgArr.shape, dtype = "uint8")
					# cv2.drawContours(outline, cnts, -1, (255, 255, 255), -1)

					tempPath = os.path.join(classAug, img)
					cv2.imwrite(tempPath, th3)


					# cv2.imshow('fdvsfdvfd', sobelx)
					# cv2.waitKey()
				

					# cv2.imwrite('lap.jpg', dst)
					# cv2.imwrite('x.jpg', sobelx)
					# cv2.imwrite('y.jpg', sobely)





			# 		imgPadd = padding(imgArr)
			# 		for i in range(4):
			# 			imgRt = rotateImg(imgPadd, rt_mat[i])
			# 			imgEros = imgErrode(imgRt)
			# 			imgDial = imgDilation(imgRt)
						
			# 			imgName = img.split('.')[0]
			# 			rtName = imgName + '_' + str(rt_mat[i]) + '.jpg'
			# 			errodeName = imgName + '_' + str(rt_mat[i]) + '_errode.jpg'
			# 			dialName = imgName + '_' + str(rt_mat[i]) + '_dilation.jpg'

			# 			rtPath = os.path.join(classAug, rtName)
			# 			errodePath = os.path.join(classAug, errodeName)
			# 			dialPath = os.path.join(classAug, dialName)

			# 			cv2.imwrite(rtPath, imgRt)
			# 			cv2.imwrite(errodePath, imgEros)
			# 			cv2.imwrite(dialPath, imgDial)

			# print('Images written for class,\t', className)