# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from YOLOv3.utils.misc_utils import parse_anchors, read_class_names
from YOLOv3.utils.nms_utils import gpu_nms
from YOLOv3.utils.plot_utils import get_color_table, plot_one_box
from YOLOv3.utils.data_aug import letterbox_resize

from YOLOv3.model import yolov3

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
args = parser.parse_args()

anchor_path = "./YOLOv3/data/yolo_anchors.txt"
new_size = [416, 416]
letterbox_isResize = True
class_name_path = "./YOLOv3/data/coco.names"
restore_path = "./YOLOv3/data/darknet_weights/yolov3.ckpt"
parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")

args.anchors = parse_anchors(anchor_path)
args.classes = read_class_names(class_name_path)
args.num_class = len(args.classes)
color_table = get_color_table(args.num_class)


def detect_vehicle(input_image):
	img_ori = cv2.imread(input_image)
	if letterbox_isResize:
		img, resize_ratio, dw, dh = letterbox_resize(img_ori, new_size[0], new_size[1])
	else:
		height_ori, width_ori = img_ori.shape[:2]
		img = cv2.resize(img_ori, tuple(new_size))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = np.asarray(img, np.float32)
	img = img[np.newaxis, :] / 255.

	with tf.Session() as sess:
		input_data = tf.placeholder(tf.float32, [1, new_size[1], new_size[0], 3], name='input_data')
		yolo_model = yolov3(args.num_class, args.anchors)
		with tf.variable_scope('yolov3'):
			pred_feature_maps = yolo_model.forward(input_data, False)
		pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

		pred_scores = pred_confs * pred_probs

		boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3,
		                                nms_thresh=0.45)

		saver = tf.train.Saver()
		saver.restore(sess, restore_path)

		boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

		# rescale the coordinates to the original image
		if letterbox_isResize:
			boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
			boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
		else:
			boxes_[:, [0, 2]] *= (width_ori / float(new_size[0]))
			boxes_[:, [1, 3]] *= (height_ori / float(new_size[1]))

		# print("box coords:")
		# print(boxes_)
		# print('*' * 30)
		# print("scores:")
		# print(scores_)
		# print('*' * 30)
		# print("labels:")
		# print(labels_)

		for i in range(len(boxes_)):
			x0, y0, x1, y1 = boxes_[i]
			plot_one_box(img_ori, [x0, y0, x1, y1],
			             label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
			             color=color_table[labels_[i]])
		# cv2.imshow('Detection result', img_ori)
		detection_result = './detection_result.jpg'
		cv2.imwrite(detection_result, img_ori)
		# cv2.waitKey(0)
		return detection_result

# if __name__ == '__main__':
# parser.add_argument("--input_image", type=str, default='./data/my_data/reverse_339_734_432_320.jpg',
#                     help="The path of the input image.")
# parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
#                     help="The path of the anchor txt file.")
# parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
#                     help="Resize the input image with `new_size`, size format: [width, height]")
# parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
#                     help="Whether to use the letterbox resize.")
# parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
#                     help="The path of the class names.")
# parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
#                     help="The path of the weights to restore.")
# args = parser.parse_args()
