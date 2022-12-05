import cv2
import skimage.filters
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from modules.utils import CLASSES, IMAGE_SIZE
from modules.m1_datapreparation import segmentation_process, padded_resize, resize_input


size = (IMAGE_SIZE['height'], IMAGE_SIZE['width'])
per_batch_size = 64

def create_datagen(canny=True, split=0):
	return ImageDataGenerator(
			preprocessing_function=preprocessing_function if canny else None,
			validation_split = split, # 0.3
			rescale = 1./255, # normalization
		)


def flow_dataset(datagen, src_path, category, subset):
	return datagen.flow_from_directory(
		directory=src_path,
		target_size=size,
		classes=CLASSES[category],
		batch_size=per_batch_size,
		color_mode='grayscale',
		shuffle=True,
		subset=subset, # training, 0.7; validation, 0.3
	)


def sloven_split(category):
	# pre-calculated slovin split
	classes_split = {
		'gender': [0.09092, 0.33345], # 9092 train, 6668 valid
		'handedness': [0.09092, 0.2858], # 9092 train, 7144 valid
		'gender-handedness': [0.07145, 0.33345], # 9288 train, 6668 valid
	}

	split1 = classes_split[category][0]
	split2 = classes_split[category][1]

	return split1, split2


def load_train_valid_tuner(path='data/wordImages', category=None, canny=True, dummy=False, training_dataset_only=False):
	train_path=f'{path}/{category}/training'
	valid_path=f'{path}/{category}/validation'

	if dummy: 
		split1 = split2 = .1
		if training_dataset_only:
			valid_path = train_path
	else:
		split1, split2 = sloven_split(category)

	datagen1 = create_datagen(canny=canny, split=split1)
	datagen2 = create_datagen(canny=canny, split=split2)

	# ---------------------------------------------------
	print('[TUNER] training dataset:', end ="\t" ) # 100,000 -> 9,092
	train_batch = flow_dataset(datagen1, train_path, category, 'validation')
	
	print('[TUNER] validation dataset:', end ="\t") # 20,000 -> 6,668
	valid_batch = flow_dataset(datagen1, valid_path, category, 'validation')

	return train_batch, valid_batch


def load_train_valid(path='data/wordImages', category=None, canny=True, dummy=False, training_dataset_only=False):
	train_path = f'{path}/{category}/training'
	valid_path = f'{path}/{category}/validation'

	if dummy: 
		split1 = split2 = .1
		subset1 = 'validation'
		subset2 = 'validation'
		if training_dataset_only:
			valid_path = train_path
	else:
		if training_dataset_only:
			split1, split2 = (0.15, 0.15)
			subset1 = 'training'
			subset2 = 'validation'
			valid_path = train_path
		else:
			split1, split2 = (0, 0)
			subset1 = None
			subset2 = None

	datagen1 = create_datagen(canny=canny, split=split1)
	datagen2 = create_datagen(canny=canny, split=split2)

	# ---------------------------------------------------
	print(f'[FULL] Training dataset:', end ="\t" )
	train_batch = flow_dataset(datagen1, train_path, category, subset1)

	# ---------------------------------------------------
	print(f'[FULL] Validation dataset:', end ="\t")
	valid_batch = flow_dataset(datagen2, valid_path, category, subset2)

	return train_batch, valid_batch

# NOT USED
def load_testing(path='data/wordImages', category=None, canny=True, batch_size=None, dummy=False):
	if not batch_size:
		batch_size=per_batch_size

	if dummy:
		test_path = f'{path}/{category}/validation'
		datagen = create_datagen(canny=canny, split=.05)
		msg = 'DUMMY'
		subset = 'validation'
	else:
		test_path = f'{path}/{category}/testing'
		datagen = create_datagen(canny=canny, split=0)
		msg = 'FULL'
		subset = None

	print(f'[{msg}] Testing dataset:', end ="\t")
	return flow_dataset(datagen, test_path, category, subset)


def load_testlines(path='data/lineSplitDataset', category=None, canny=True, batch_size=None, dummy=False):
	if not batch_size:
		batch_size=per_batch_size

	datagen = create_datagen(canny=canny)
	msg = 'DUMMY' if dummy else 'FULL'
	test_x = []
	test_y = []

	for class_num, perclass in enumerate(CLASSES[category]):
		print(f'CLASS: {perclass}')
		src_path = f'{path}/{category}/test/{perclass}'

		lines = segmentation_process(src_path=src_path) # segmentation

		print('Resizing.')
		resized_word_lines = []
		for line in lines:
			resized_line_n = []
			for word in line:
				resized_line_n.append(padded_resize(word, size))
				# img = resize_input(word)
				# print(img.shape, end='\r')
				# resized_line_n.append(img)
			if resized_line_n == []: continue
			resized_word_lines.append(resized_line_n)

		print('Preprocessing.')
		test_words_x_perclass = []
		test_words_y_perclass = []
		for line in resized_word_lines:
			test_line_x = np.array(line)
			test_line_y = np.array([perclass] * len(line))

			data_perword = datagen.flow(
				x=test_line_x,
				y=test_line_y,
			)

			preprocessed_test_words_x = []
			for i in range(len(test_line_x)):
				img_word, lbl = data_perword.next()
				# preprocessed_test_words_x.append(np.array(img_word))
				preprocessed_test_words_x.append(tf.image.rgb_to_grayscale(img_word[0]).numpy()[np.newaxis,:,:,:])

			test_words_x_perclass.append(preprocessed_test_words_x)
			test_words_y_perclass.append(class_num)

		test_x += test_words_x_perclass
		test_y += test_words_y_perclass

		print()

	return test_x, test_y


def preprocessing_function(img):
	
	img = img.astype("uint8")

	# apply canny edge detection with automatic thresholding
	canny = cv2.Canny(img,100,200)


	# canny shape: (30, 100, 1)

	# add color channel to binarized image
	gray = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
	# gray shape: (30, 100, 3)

	preprocessed_img = tf.image.rgb_to_grayscale(gray) # just adds batch dimension
	# tf shape: (1, 30, 100, 1)

	preprocessed_img = preprocessed_img.numpy() / 1
	# numpy shape: (1, 30, 100, 1) just in numpy format

	return preprocessed_img