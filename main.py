#encoding:utf-8
'''
=======================================================================================
Author: Chun-Wei Chiang
Date: 2019.02.24
Description: Train recognize license plate model
=======================================================================================
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import model
import glob
import os
import cv2 
from tensorflow.keras.utils import to_categorical

provNum, alphaNum, adNum = 38, 25, 35
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
img_origin_width = 720
img_origin_height = 1160
img_origin_channel = 3


def buildLabel(img_path):
	#([x,y,w,h], [y0, y1, y2, y3, y4, y5, y6])
	img_name = img_path.split('/')[-1].rsplit('.',1)[0]
	label_array = img_name.split('-')
	
	# bbox location
	[lefttop , rightdown] = label_array[2].split('_')
	[lefttop_x, lefttop_y] = [float(i) for i in lefttop.split('&')]
	[rightdown_x, rightdown_y] = [float(i) for i in rightdown.split('&')]

	# normalize the pixel to 0 ~ 1
	lefttop_x , rightdown_x = lefttop_x/img_origin_width, rightdown_x/img_origin_width
	lefttop_y , rightdown_y = lefttop_y/img_origin_height ,rightdown_y/img_origin_height

	# plate number 
	plate_number = label_array[4].split('_')
	plate_number[0] = to_categorical(plate_number[0],num_classes=provNum)
	plate_number[1] = to_categorical(plate_number[1],num_classes=alphaNum)
	plate_number[2] = to_categorical(plate_number[2],num_classes=adNum)
	plate_number[3] = to_categorical(plate_number[3],num_classes=adNum)
	plate_number[4] = to_categorical(plate_number[4],num_classes=adNum)
	plate_number[5] = to_categorical(plate_number[5],num_classes=adNum)
	plate_number[6] = to_categorical(plate_number[6],num_classes=adNum)


	# print(plate_number)

	return np.array([ (lefttop_x+rightdown_x)/2, (lefttop_y+rightdown_y)/2, rightdown_x-lefttop_x, rightdown_y-lefttop_y ]), plate_number

def loadData(store_path = 'ccpd_dataset/ccpd_base/'):
	img_list = []
	box_labels = []
	labels_0 = []
	labels_1 = []
	labels_2 = []
	labels_3 = []
	labels_4 = []
	labels_5 = []
	labels_6 = []
	
	counter = 0

	for img_path in glob.glob( store_path + '*.jpg'):
		if counter > 5000:
			break
		img_array = cv2.imread( img_path )
		img_array = cv2.resize( img_array, (180, 290))
		img_list.append(img_array) 
		box_label, lp_label = buildLabel(img_path)	
		box_labels.append(box_label)	
		labels_0.append(lp_label[0])	
		labels_1.append(lp_label[1])	
		labels_2.append(lp_label[2])	
		labels_3.append(lp_label[3])	
		labels_4.append(lp_label[4])	
		labels_5.append(lp_label[5])	
		labels_6.append(lp_label[6])	
			
		#Test for viewimg data
		# plt.imshow(img_array)
		# plt.show()

		counter += 1

	img_list = np.array(img_list)
	box_labels = np.array(box_labels)
	labels_0 = np.array(labels_0)
	labels_1 = np.array(labels_1)
	labels_2 = np.array(labels_2)
	labels_3 = np.array(labels_3)
	labels_4 = np.array(labels_4)
	labels_5 = np.array(labels_5)
	labels_6 = np.array(labels_6)

	img_list = img_list/255.0
	# print(img_label)
	return img_list, box_labels, labels_0, labels_1, labels_2, labels_3, labels_4, labels_5, labels_6

if __name__ == '__main__':
	img_data, box_labels, labels_0, labels_1, labels_2, labels_3, labels_4, labels_5, labels_6 = loadData()

	with tf.Session() as sess:
		tf.set_random_seed(1)

		RLPRnet = model.trainModel(img_data, box_labels, labels_0, labels_1, labels_2, labels_3, labels_4, labels_5, labels_6)