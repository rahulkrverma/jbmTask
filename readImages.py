


from random import shuffle
import glob
import sys
import cv2
import numpy as np
import tensorflow as tf

def getLabel(filePaths):
	labels = []
	for img in filePaths:
		if 'SC' in img:
			labels.append(0)
		elif 'SD' in img:
			labels.append(1)
		elif 'TH' in img:
			labels.append(2)
		elif 'WR' in img:
			labels.append(3)
		elif 'Back' in img:
			labels.append(4)
		elif 'Front' in img:
			labels.append(5)

	dataZip = list(zip(filePaths, labels))
	shuffle(dataZip)
	filePaths, labels = zip(*dataZip)
	return filePaths, labels

def load_image(filePath):
	img = cv2.imread(filePath)
	if img is None:
		return None
	img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def createDataRecord(out_filename, filePaths, labels):
	writer = tf.python_io.TFRecordWriter(out_filename)
	for x in range(len(filePaths)):
		img = load_image(filePaths[x])
		label = labels[x]
		if img is None:
			continue

		feature = {
			'image_raw': _bytes_feature(img.tostring()),
			'label': _int64_feature(label)
		}

		example = tf.train.Example(features=tf.train.Features(feature=feature))
		writer.write(example.SerializeToString())	
	writer.close()
	sys.stdout.flush()


if __name__ == '__main__':

	trainPath = '../All_61326/train_61326/*/*'
	testPath = '../All_61326/test_61326/*'
	
	trainImgs = glob.glob(trainPath)
	testImgs = glob.glob(testPath)

	trainImgs, trainLabel = getLabel(trainImgs)
	testImgs, testLabel = getLabel(testImgs)

	createDataRecord('../data/train.tfrecords', trainImgs, trainLabel)
	createDataRecord('../data/test.tfrecords', testImgs, testLabel)



