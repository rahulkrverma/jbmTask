

import os
import cv2
import numpy as np
import tensorflow as tf


def get_model_filenames(model_dir):
	files = os.listdir(model_dir)
	meta_files = [s for s in files if s.endswith('.meta')]
	if len(meta_files)==0:
		raise ValueError('No meta file found in the model directory (%s)' % model_dir)
	elif len(meta_files)>1:
		raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
	meta_file = meta_files[0]
	ckpt = tf.train.get_checkpoint_state(model_dir)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
		return meta_file, ckpt_file

	meta_files = [s for s in files if '.ckpt' in s]
	max_step = -1
	for f in files:
		step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
		if step_str is not None and len(step_str.groups())>=2:
			step = int(step_str.groups()[1])
			if step > max_step:
				max_step = step
				ckpt_file = step_str.groups()[0]
	return meta_file, ckpt_file


def load_model(img, model = '../loadModel'):
	# saver = tf.train.import_meta_graph('../loadModel/model.ckpt-140.meta')
	# saver.restore(tf.get_default_session(), '../loadModel/model.ckpt-140')

	# model_exp = os.path.expanduser(model)
	# if (os.path.isfile(model_exp)):
	# 	print('Model filename: %s' % model_exp)
	# 	with gfile.FastGFile(model_exp,'rb') as f:
	# 		graph_def = tf.GraphDef()
	# 		graph_def.ParseFromString(f.read())
	# 		tf.import_graph_def(graph_def, name='')
	# else:
	# 	print('Model directory: %s' % model_exp)
	# 	meta_file, ckpt_file = get_model_filenames(model_exp)
		
	# 	print('Metagraph file: %s' % meta_file)
	# 	print('Checkpoint file: %s' % ckpt_file)
	
	# 	saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
	# 	saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
	
	# tt = tf.saved_model.loader.load(export_dir = '../model', sess = None, tags = None)
	
	# saver = tf.train.import_meta_graph('../loadModel/model.ckpt-140.meta')
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('../loadModel/model.ckpt-140.meta')
		saver.restore(sess, tf.train.latest_checkpoint('../loadModel/'))
		# kk = sess.run(img)


		graph = tf.get_default_graph()
		# feed_dict = {'output_cls' : img}

		# kk = sess.run(feed_dict)

		# feed_dict = {x: img}
		# classification = tf.run(y, feed_dict)

		images_placeholder = tf.get_default_graph().get_tensor_by_name("input_tensor_after:0")
		embeddings = tf.get_default_graph().get_tensor_by_name("output_cls:0")
		# phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			# Run forward pass to calculate embeddings
		# y = tf.nn.softmax(logits = graph)
		feed_dict = { images_placeholder: img}
		emb = sess.run(embeddings, feed_dict=feed_dict)

		print(emb)






if __name__ == '__main__':
	print('functin called')

	im = []

	img = cv2.imread('../All_61326/test_61326/61326-WR- (20).jpg')
	img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# img = tf.cast(img, tf.float32)
	im.append(img)
	# img_1 = np.stack(img)

	load_model(im)



















