#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import time
import uuid
import tensorflow as tf
from flask import Flask, redirect, request, send_from_directory, url_for
# from inference import run_inference_on_image
# from classify_image import run_inference_on_image
from PIL import Image

print('current working dir [{0}]'.format(os.getcwd()))
w_d = os.path.dirname(os.path.abspath(__file__))
print('change wording dir to [{0}]'.format(w_d))
os.chdir(w_d)
 
from YOLOv3 import vehicle_detect_yolo
import classify_image



ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG', 'png'])

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('model_dir', '', """Path to graph_def pb, """)
# tf.app.flags.DEFINE_string('model_name_detect', 'my_inception_v4_freeze.pb', '')
# tf.app.flags.DEFINE_string('label_file_detect', 'my_inception_v4_freeze.label', '')
# tf.app.flags.DEFINE_string('model_name_recog', './freezed.pb', '')
# tf.app.flags.DEFINE_string('label_file_recog', './freezed.label', '')
tf.app.flags.DEFINE_string('upload_folder', '/tmp/', '')
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")
tf.app.flags.DEFINE_integer('port', '5001',
                            'server with port,if no port, use deault port 80')

tf.app.flags.DEFINE_boolean('debug', False, '')

UPLOAD_FOLDER = FLAGS.upload_folder
ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG', 'png'])

app = Flask(__name__)
app._static_folder = UPLOAD_FOLDER


def allowed_files(filename):
	return '.' in filename and \
	       filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def rename_filename(old_file_name):
	basename = os.path.basename(old_file_name)
	name, ext = os.path.splitext(basename)
	new_name = str(uuid.uuid1()) + ext
	return new_name


def vehicle_inference(file_name):
	# 车辆检测——YOLO v3
	try:
		output_img, bbox = vehicle_detect_yolo.detect(file_name)
	except Exception as ex:
		print(ex)
		return ex

	if output_img is not None:
		image = Image.open(output_img)
		# cropped_image = image.crop([bbox[0], bbox[1], bbox[2], bbox[3]])
		# cropped_result = './cropped_image.jpg'
		# cropped_image.save(cropped_result, quality=95, subsampling=0)

		filename = rename_filename(output_img)
		file_path = os.path.join(UPLOAD_FOLDER, filename)
		image.save(file_path)

		new_url = '/static/%s' % os.path.basename(file_path)
		image_tag = '<img src="%s"></img><p>'
		new_tag = image_tag % new_url

		# 车辆分类——Inception v4
		try:
			predictions, top_k, top_names = classify_image.run_inference_on_image1(file_name)
		except Exception as ex:
			print(ex)
			return ex

		format_string = ''
		for node_id, human_name in zip(top_k, top_names):
			score = predictions[node_id]
			format_string += 'id:[%d] name:[%s] (score = %.5f)<BR>' % (node_id, human_name, score)
	else:
		new_url = '/static/%s' % os.path.basename(file_name)
		image_tag = '<img src="%s"></img><p>'
		new_tag = image_tag % new_url
		format_string = '未检测到车辆信息！'

	# 返回检测及分类结果
	ret_string = new_tag + format_string + '<BR>'
	return ret_string


@app.route("/", methods=['GET', 'POST'])
def root():
	result = """
    <!doctype html>
    <title>车辆检测及型号识别系统</title>
    <h1>请选择一张车辆图片</h1>
    <form action="" method=post enctype=multipart/form-data>
         <input type=file name=file value='选择图片' style='width:300px'>
         <input type=submit value='上传'>
    </form>
    <p>%s</p>
    """ % "<br>"
	if request.method == 'POST':
		file = request.files['file']
		old_file_name = file.filename
		if file and allowed_files(old_file_name):
			filename = rename_filename(old_file_name)
			file_path = os.path.join(UPLOAD_FOLDER, filename)
			file.save(file_path)
			type_name = 'N/A'
			print('file saved to %s' % file_path)
			out_html = vehicle_inference(file_path)
			return result + out_html
	return result


if __name__ == "__main__":
	print('listening on port %d' % FLAGS.port)
	app.run(host='0.0.0.0', port=FLAGS.port, debug=FLAGS.debug, threaded=True)
