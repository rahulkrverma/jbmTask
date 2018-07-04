

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
	augPath = '/home/user/Desktop/jbmTask/imgAug'

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
					imgArr = cv2.resize(imgArr, (224, 224), interpolation=cv2.INTER_CUBIC)

					imgPadd = padding(imgArr)
					for i in range(4):
						imgRt = rotateImg(imgPadd, rt_mat[i])
						imgEros = imgErrode(imgRt)
						imgDial = imgDilation(imgRt)
						
						imgName = img.split('.')[0]
						rtName = imgName + '_' + str(rt_mat[i]) + '.jpg'
						errodeName = imgName + '_' + str(rt_mat[i]) + '_errode.jpg'
						dialName = imgName + '_' + str(rt_mat[i]) + '_dilation.jpg'

						rtPath = os.path.join(classAug, rtName)
						errodePath = os.path.join(classAug, errodeName)
						dialPath = os.path.join(classAug, dialName)

						cv2.imwrite(rtPath, imgRt)
						cv2.imwrite(errodePath, imgEros)
						cv2.imwrite(dialPath, imgDial)

			print('Images written for class,\t', className)