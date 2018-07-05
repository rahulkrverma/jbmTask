
import cv2
import numpy as np
import tensorflow as tf

from img_preProcess import preProcess

def predict_label(img):
	img = preProcess_img(img)
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('../model/model.ckpt-140.meta')
		saver.restore(sess, tf.train.latest_checkpoint('../model/'))
		graph = tf.get_default_graph()

		inputTensor_name = tf.get_default_graph().get_tensor_by_name("input_tensor_after:0")
		output_name = tf.get_default_graph().get_tensor_by_name("output_cls:0")
		
		feed_dict = {inputTensor_name : img}
		predLabel = sess.run(output_name, feed_dict = feed_dict)
		
		return predLabel

def preProcess_img(imgArr):
	imgArr = cv2.resize(imgArr, (224, 224), interpolation = cv2.INTER_CUBIC)
	imgArr = cv2.cvtColor(imgArr, cv2.COLOR_BGR2RGB)

	imgArr = preProcess(imgArr)
	
	return [imgArr]
