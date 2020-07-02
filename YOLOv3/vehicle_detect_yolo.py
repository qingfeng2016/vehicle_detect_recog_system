# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import random
import os

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

args.anchors = parse_anchors(anchor_path)
args.classes = read_class_names(class_name_path)
args.num_class = len(args.classes)
color_table = get_color_table(args.num_class)

print('正在加载YOLOV3检测模型，请稍候……')
# with tf.Session() as sess:
sess = tf.Session()
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


def detect(input_image):
	img_ori = cv2.imread(input_image)
	if letterbox_isResize:
		img, resize_ratio, dw, dh = letterbox_resize(img_ori, new_size[0], new_size[1])
	else:
		height_ori, width_ori = img_ori.shape[:2]
		img = cv2.resize(img_ori, tuple(new_size))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = np.asarray(img, np.float32)
	img = img[np.newaxis, :] / 255.

	boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

	# rescale the coordinates to the original image
	if letterbox_isResize:
		boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
		boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
	else:
		boxes_[:, [0, 2]] *= (width_ori / float(new_size[0]))
		boxes_[:, [1, 3]] *= (height_ori / float(new_size[1]))

	return bboxes_filter(img_ori, boxes_, scores_, labels_)


def bboxes_filter(img_ori, bboxes, scores, labels):
	'''
	如多个bboxes，只保留一个bbox;
	bbox筛选条件：面积最大，位置居中，得分大于0.3
	'''

	idxs = []
	areas = []

	for i, bbox in enumerate(bboxes):
		if float(scores[i]) < 0.3 or labels[i] not in (2, 3, 5, 6, 7):
			continue
		# 计算bbox面积
		box_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
		areas.append(box_area)
		idxs.append(i)

	# 被排除的图片（没有检测到车辆，或者分数全部小于0.3）
	if (len(idxs) == 0):
		return None, None

	# 筛选出面积最大的bbox索引
	idx_keep = idxs[0]
	if (len(idxs) > 1):
		index = np.argsort(-np.array(areas))
		idxArray = np.array(idxs)[index]
		idx_keep = idxArray[0]

	print("box coord: " + str(bboxes[idx_keep]))
	print("score: " + str(scores[idx_keep]))
	print("label: " + str(labels[idx_keep]))

	# 添加bbox
	x0, y0, x1, y1 = bboxes[idx_keep]
	# plot_one_box(img_ori, [x0, y0, x1, y1],
	#              label=args.classes[labels[idx_keep]] + ', {:.2f}%'.format(scores[idx_keep] * 100),
	#              color=color_table[labels[idx_keep]])
	plot_one_box(img_ori, [x0, y0, x1, y1],
	             label=args.classes[2]+', {:.2f}%'.format(scores[idx_keep] * 100),
	             color=[0,191,255])
	detection_result = './detection_result.jpg'
	cv2.imwrite(detection_result, img_ori)
	return detection_result, bboxes[idx_keep]


def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    # tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    tl = 1
    # color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

