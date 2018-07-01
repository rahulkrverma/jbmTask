
from flask import Flask
from flask import Flask, request, jsonify

import numpy as np
import cv2
import io

import predict

app = Flask(__name__)

@app.route('/getLabel', methods = ['GET', 'POST'])
def getImage():
	try:
		if 'img' in request.files:
			fileObj = request.files['img']
			in_memory_file = io.BytesIO()
			fileObj.save(in_memory_file)
			imgObj = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
			color_image_flag = 1
			imgArr = cv2.imdecode(imgObj, color_image_flag)
			imgLabel = str((predict.predict_label(imgArr))[0])

			ret = {'Label' : imgLabel,
					'status' : 1,
					'msg' : '‘success’'}
			return jsonify(ret)
		else:
			ret = {'Label' : None,
					'status' : 0,
					'msg' : 'wrong header name'}
			return jsonify(ret)
	except Exception as e:
		err = str(e)
		ret = {'Label' : None,
				'status' : 0,
				'msg' : err}
		return jsonify(ret)

if __name__ == '__main__':
	app.run(debug = True, host = '0.0.0.0')