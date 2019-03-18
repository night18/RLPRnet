'''
=======================================================================================
Author: Chun-Wei Chiang
Date: 2019.03.16
Description: Set the model to recognize license plate
=======================================================================================
Change logs
2019.03.16 [Chun] 
=======================================================================================
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Input, Conv2D, Activation, concatenate, MaxPool2D, Flatten, Dense, BatchNormalization, ZeroPadding2D, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Lambda
from pprint import pprint
import pickle


models_dir = "models"
history_dir = "history"
checkpoint_dir = "checkpoint"
box_in_x =None
dropout_rate = 0.5

def shortcutModule(pre_lyr, output_channels, res_id, is_stride = True):
	if is_stride:
		shortcut = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same', name=res_id+'_shortcut_pool')(pre_lyr)
	else:
		shortcut = pre_lyr

	shortcut = Conv2D(output_channels, kernel_size=(1,1), padding='same', name=res_id+'shortcut_conv')(shortcut)
	shortcut = BatchNormalization(name=res_id+'shortcut_norm')(shortcut)
	return shortcut



def myResModule(pre_lyr, kernel_length, first_channels, second_channels, res_id):
	x = Conv2D(first_channels, kernel_size=(kernel_length,kernel_length), strides=(2,2) ,padding='same', name=res_id+'_conv_1' )(pre_lyr)
	x = BatchNormalization(name=res_id+'_batch_1')(x)
	x = Dropout(dropout_rate, name=res_id+'_dropout_1')(x)
	x = Activation('relu', name=res_id+'_relu_1')(x)

	x = Conv2D(second_channels, kernel_size=(kernel_length,kernel_length), strides=(1,1), padding='same', name=res_id+'_conv_2')(x)
	x = BatchNormalization(name=res_id+'_batch_2')(x)

	shortcut = shortcutModule(pre_lyr, second_channels, res_id)
	x = concatenate([x, shortcut], name=res_id+'_conc')
	x = Dropout(dropout_rate, name=res_id+'_dropout_2')(x)
	x = Activation('relu', name=res_id+"_relu_2")(x)
	return x

def roiPooling(pre_lyr, pool_height=8, pool_weight=16):
	global box_in_x
	rois = box_in_x
	box_ind = tf.cast(rois[:,0],dtype=tf.int32)
	boxes = rois[:,1:]
	crop_size = tf.constant([2*pool_height,2*pool_weight])
	pooledFeature = tf.image.crop_and_resize(image=pre_lyr, boxes=boxes, box_ind= box_ind, crop_size=crop_size, name= 'roicrop')
	pooledFeature = MaxPool2D(pool_size=(2,2), strides=(2,2), name = 'roipool', padding='same')(pooledFeature)
	return pooledFeature

def charClassifier(pre_lyr, class_num, class_idx):
	x = Flatten()(pre_lyr)
	x = Dense(128, name= class_idx + '_cls_fc_1')(x)
	x = BatchNormalization(name=class_idx+'_cls_batch_1')(x)
	x = Activation('relu', name=class_idx+'_cls_relu_1')(x)

	x = Dense(class_num, name= class_idx + '_cls_fc_2')(x)
	x = BatchNormalization(name=class_idx+'_cls_batch_2')(x)
	prediction = Activation('softmax', name=class_idx+'_cls_output')(x)

	return prediction

def RLPRnet(imput_img, input_width = 180, input_height = 290): #TODO make width to global
	global box_in_x

	inputs = Input(shape=(input_height,input_width,3))
	box_output = tf.placeholder(tf.float32, shape=[None, 4], name='box_output')

	x = Conv2D(64, kernel_size=(7,7), strides=(2,2) ,padding='same', name= '0_conv_1' )(inputs)
	x = myResModule(x, 5, 64, 128, "1")
	x = myResModule(x, 5, 128, 192, "2")
	# x = myResModule(x, 5, 192, 192, "3")

	# Box location
	x = myResModule(x, 3, 192, 192, "box")
	# box = Conv2D(192, kernel_size=(3,3), strides=(2,2), padding='same', name = 'box_conv_1')(x)
	# box = Conv2D(192, kernel_size=(3,3), strides=(1,1), padding='same', name = 'box_conv_2')(box)
	box = Flatten()(x)
	box = Dense(100, activation = 'relu', name='box_fc_1')(box)
	box = Dense(100, activation = 'relu', name='box_fc_2')(box)
	# x, y, w, h
	box_output = Dense(4, activation = 'sigmoid', name='box_classifier')(box)

	#transfer x, y, w, h to x1, y1, x2, y2
	postfix = tf.constant(np.array([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]),dtype=tf.float32 , name="postfix")
	box_new = tf.clip_by_value(tf.tensordot(box_output, postfix, axes = 1), 0, 1)

	# (samples, new_rows, new_cols, filters) https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
	# x_h = tf.shape(x)[1]
	# x_w = tf.shape(x)[2]
	x_h = 1
	x_w = 1
	# transfer to y1, x1, y2, x2
	fix_in_x = tf.constant(np.array([[0,0,x_h,0,0],[0,x_w,0,0,0],[0,0,0,0,x_h],[0,0,0,x_w,0]]),dtype=tf.float32 , name="fix_in_x")
	box_in_x = tf.tensordot(box_new, fix_in_x, axes = 1)
	roi = Lambda(roiPooling)(x)

	# Plate Recognition
	# pr = Conv2D(192, kernel_size=(3,3), strides=(2,2), padding='same', name = 'box_conv_1')(x)
	# pr = Conv2D(192, kernel_size=(3,3), strides=(1,1), padding='same', name = 'box_conv_1')(pr)

	# y0 = charClassifier(roi,38 ,"0")
	# y1 = charClassifier(roi,25 ,"1")
	y2 = charClassifier(roi,35 ,"2")
	y3 = charClassifier(roi,35 ,"3")
	y4 = charClassifier(roi,35 ,"4")
	y5 = charClassifier(roi,35 ,"5")
	y6 = charClassifier(roi,35 ,"6")

	# outputs = concatenate([box_output, y0, y1, y2, y3, y4, y5, y6],axis=1, name='final_output')

	model = Model(
		inputs = inputs,
		# outputs = [box_output, y0, y1, y2, y3, y4, y5, y6], #tuple
		outputs = [box_output, y2, y3, y4, y5, y6], #tuple
		name = 'RLPRnet'
		)
	return model

def customLoss(yTrue, yPred):
	pprint(yTrue.get_shape())
	pprint(yPred.get_shape())
	#([x,y,w,h], [y0, y1, y2, y3, y4, y5, y6])
	true_box = yTrue[0]
	true_lpn = yTrue[1]
	pred_box = yPred[0]
	pred_lpn = yPred[1]

	bbox_loss = 0
	lpn_loss = 0

	if true_box != None:
		for idx, x in enumerate(true_box):
			if abs(x - pred_box[idx]) > 1:
				bbox_loss += abs(x - pred_box[idx]) - 0.5
			else:
				bbox_loss += 0.5 * (x - pred_box[idx])**2

	if true_box != None:
		for idx, x in enumerate(true_lpn):
			lpn_loss += K.categorical_crossentropy(pred_lpn[idx] ,x)

	loss = tf.reduce_sum(bbox_loss + lpn_loss)
	pprint(loss.get_shape())
	return loss

# def bboxLoss(yTrue, yPred):
# 	bbox_loss = 0
# 	K.sum()
# 	if yTrue != None:
# 		for idx, x in enumerate(yTrue):
# 			if abs(x - yPred[idx]) > 1:
# 				bbox_loss += abs(x - yPred[idx]) - 0.5
# 			else:
# 				bbox_loss += 0.5 * (x - yPred[idx])**2
# 	return 

#Custom metric
# def IOUmetric(y_true, y_pred):
# 	# [x,y,w,h]
# 	union = (y_pred[0]-y_pred[2]) - (y_true[0] + y_true[1])

# 	print(y_true)
# 	return K.cast(0, 'int32')

def trainModel(train_data, box_labels, labels_0, labels_1, labels_2, labels_3, labels_4, labels_5, labels_6, epochs=100, learning_rate=0.001):
	model = None
	h5_storage_path = models_dir + "/" + "RLPRnet_" + str(learning_rate) + ".h5"
	hist_storage_path = history_dir + "/" + "RLPRnet_" + str(learning_rate)
	checkpoint_path = checkpoint_dir + "/" + "RLPRnet_" + str(learning_rate) + ".hdf5"

	model = RLPRnet(train_data)
	losses = {
		"box_classifier": "mse", 
		# "0_cls_output": "categorical_crossentropy",
		# "1_cls_output": "categorical_crossentropy",
		"2_cls_output": "categorical_crossentropy",
		"3_cls_output": "categorical_crossentropy",
		"4_cls_output": "categorical_crossentropy",
		"5_cls_output": "categorical_crossentropy",
		"6_cls_output": "categorical_crossentropy",
	}

	loss_weights = {
		"box_classifier": 0.5, 
		# "0_cls_output": 0.1,
		# "1_cls_output": 0.1,
		"2_cls_output": 1,
		"3_cls_output": 1,
		"4_cls_output": 1,
		"5_cls_output": 1,
		"6_cls_output": 1,
	}
	#TODO metrics
	model.compile(
		loss = losses,
		loss_weights=loss_weights,
		optimizer = Adam(lr=learning_rate),
		metrics = ["accuracy"]
	)

	checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	#Fit the model
	hist = model.fit(
		train_data,
		{
		"box_classifier": box_labels,
		# "0_cls_output": labels_0,
		# "1_cls_output": labels_1,
		"2_cls_output": labels_2,
		"3_cls_output": labels_3,
		"4_cls_output": labels_4,
		"5_cls_output": labels_5,
		"6_cls_output": labels_6,
		},
		epochs = epochs,
		batch_size = 32,
		validation_split = 0.3,
		callbacks=callbacks_list,
		verbose= 1)

	#save the model
	save_model(
		model,
		h5_storage_path,
		overwrite=True,
		include_optimizer=True
	)

	#Save the history of training
	with open(hist_storage_path, 'wb') as file_hist:
		pickle.dump(hist.history, file_hist)

	print("Successfully save the model at " + h5_storage_path)

	return model


def loadModel(learning_rate = 0.001):
	h5_storage_path = models_dir + "/" + "RLPRnet_" + str(learning_rate) + ".h5"
	
	try:
		model = load_model(
			h5_storage_path,
			custom_objects=None,
			compile=True
		)

	except Exception as e:
		model = None
		print(e)
	finally:
		return model