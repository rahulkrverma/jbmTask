

import tensorflow as tf
import cv2
import sys, os
import numpy as np

def read_imgs(record):
	keys_to_features = {
		"image_raw": tf.FixedLenFeature([], tf.string),
		"label":     tf.FixedLenFeature([], tf.int64)
	}
	parsed = tf.parse_single_example(record, keys_to_features)
	image = tf.decode_raw(parsed["image_raw"], tf.uint8)
	image = tf.cast(image, tf.float32)
	label = tf.cast(parsed["label"], tf.int32)

	return {'image': image}, label

def input_fn(filenames):
	dataset = tf.data.TFRecordDataset(filenames=filenames)
	dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1024, 1))
	dataset = dataset.apply(tf.contrib.data.map_and_batch(read_imgs, 32))
	dataset = dataset.prefetch(buffer_size=2)
	return dataset

def train_input_fn():
	return input_fn(filenames = ["../data/train.tfrecords"])

def val_input_fn():
	return input_fn(filenames = ["../data/test.tfrecords"])

def model_fn(features, labels, mode, params):
	num_classes = 6
	net = features["image"]

	net = tf.identity(net, name = "input_tensor")
	
	net = tf.reshape(net, [-1, 224, 224, 3])    

	net = tf.identity(net, name = "input_tensor_after")

	net = tf.layers.conv2d(inputs = net, name = 'layer_conv1',
							filters = 32, kernel_size = 3,
							padding = 'same', activation = tf.nn.relu)
	net = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 3)

	net = tf.layers.conv2d(inputs = net, name = 'layer_conv2',
							filters = 64, kernel_size = 3,
							padding = 'same', activation = tf.nn.relu)
	net = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 3)  

	net = tf.layers.conv2d(inputs = net, name = 'layer_conv3',
							filters = 64, kernel_size = 3,
							padding = 'same', activation = tf.nn.relu)
	net = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 3)    

	net = tf.contrib.layers.flatten(net)

	net = tf.layers.dense(inputs = net, name = 'layer_fc1',
							units = 128, activation = tf.nn.relu)  
	
	net = tf.layers.dropout(net, rate = 0.5, noise_shape = None, 
							seed = None, training = (mode == tf.estimator.ModeKeys.TRAIN))
	
	net = tf.layers.dense(inputs = net, name = 'layer_fc_2',
							units = num_classes)

	logits = net
	y_pred = tf.nn.softmax(logits = logits)

	y_pred = tf.identity(y_pred, name = "output_pred")

	y_pred_cls = tf.argmax(y_pred, axis = 1)

	y_pred_cls = tf.identity(y_pred_cls, name = "output_cls")


	if mode == tf.estimator.ModeKeys.PREDICT:
		spec = tf.estimator.EstimatorSpec(mode = mode, predictions = y_pred_cls)
	else:
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
		loss = tf.reduce_mean(cross_entropy)

		optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		metrics = {"accuracy": tf.metrics.accuracy(labels, y_pred_cls)}

		spec = tf.estimator.EstimatorSpec(
			mode = mode,
			loss = loss,
			train_op = train_op,
			eval_metric_ops = metrics)
		
	return spec

if __name__ == '__main__':

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	model = tf.estimator.Estimator(model_fn = model_fn,
									params = {"learning_rate": 1e-4},
									model_dir = "../model/")

	count = 0
	while (count < 1):
		model.train(input_fn=train_input_fn, steps=10)
		result = model.evaluate(input_fn=val_input_fn)
		print(result)
		# print("Classification accuracy: {0:.2%}".format(result["accuracy"]))
		sys.stdout.flush()
		count = count + 1
		
