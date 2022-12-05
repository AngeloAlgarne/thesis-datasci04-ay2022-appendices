import csv
import itertools
import cv2
import os
import shutil
import glob
import numpy as np
import random
import math
import imutils
import lxml.etree
import sqlite3 as sql
import pandas as pd
import tensorflow as tf
from modules.utils import CLASSES as classes_dict, IMAGE_SIZE, _save_json, _load_json
from os import walk
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


### misc functions

## make one directory
# path 	- the path to make the dir on
# one 	- either 'train', 'training', 'test', 'testing', or 'validation'
def make_one_dir(path, one, category=None):
	orig_path = os.getcwd()
	if not os.path.isdir(path): # if path folder does not exist
		os.makedirs(path) # make folder for it
	os.chdir(path) # go to path folder 
	for categ, classes in classes_dict.items():
		if category and categ != category:
			continue

		if not os.path.isdir(categ):
			os.makedirs(categ) # make folder for the category
		os.chdir(categ) # go to the folder

		if not os.path.isdir(one):
			os.makedirs(one) # make folder for the tvt
		os.chdir(one) # go to the folder

		for perclass in classes: # make folders for each class
			if not os.path.isdir(perclass):
				os.makedirs(perclass)
		os.chdir('../../') # back to previous dir
	os.chdir(orig_path) # back to main dir

## make multiple directories
# 
def make_dirs(path, train_test=False, train_valid=False, enable_counter=True, one=None, category=None):
	if one:
		make_one_dir(path, one, category=category)
		return True

	counter = 0
	orig_path = os.getcwd()
	while True:
		#make folders if nonexistent
		if not os.path.isdir(path): # if path folder does not exist
			os.makedirs(path) # make folder for it
		else: # if path folder does exist
			print(f'* the directory \"{path}\" already exists. please delete the folder first, then retry.') # retry?
			inp = input('* do you want to retry making the directory? Y to retry, any character to terminate: ')
			if inp == 'Y' or inp == 'y': # retry
				counter += 1
				if counter == 5:
					print('You have retried 5 times already.')
					if enable_counter:
						print('System terminated.')
						return False
				continue
			else: # no retry
				return False
		os.chdir(path) # go to path folder 
		for categ, classes in classes_dict.items():
			os.makedirs(categ) # make folder for the category
			os.chdir(categ) # go to the folder

			if train_test or train_valid:
				trainstr = 'training' if train_valid else 'train'
				os.makedirs(trainstr) # make folder training
				os.chdir(trainstr) # go to the folder
				for perclass in classes: # make folders for each class
					os.makedirs(perclass)
				os.chdir('../') # back to previous dir

				if train_test:
					os.makedirs('test') # make folder testing
					os.chdir('test') # go to the folder
					for perclass in classes: # make folders for each class
						os.makedirs(perclass)
					os.chdir('../') # back to previous dir
				elif train_valid:
					os.makedirs('validation') # make folder training
					os.chdir('validation') # go to the folder
					for perclass in classes: # make folders for each class
						os.makedirs(perclass)
					os.chdir('../') # back to previous dir
			else:
				for perclass in classes: # make folders for each class
					os.makedirs(perclass)

			os.chdir('../') # back to previous dir
		os.chdir(orig_path) # back to main dir
		return True

def format_size(bytes):
    try:
        bytes = float(bytes)
        kb = bytes / 1024
    except:
        return "Error"
    if kb >= 1024:
        M = kb / 1024
        if M >= 1024:
            G = M / 1024
            return "%.2fG" % (G), G, 'G'
        else:
            return "%.2fM" % (M), M, 'M'
    else:
        return "%.2fkb" % (kb), kb, 'kb'

def copypaste_testlines(src_path, save_path):
	for i, fpath in enumerate(glob.glob(f'{src_path}/*.tif')):
		cv2.imwrite(f'{save_path}/{i}.tif', cv2.imread(fpath))
		print(f'{i+1} test images copied', end='\r')

### PHASE 1 - split according to class
# lineImages => lineSplitDataset
def make_writing_info(path='data/metadata', xml_path='writers.xml', forms_path='forms.txt'):
	if os.path.isfile(f'{path}/writing_info.csv'):
		return

	orig_path = os.getcwd()
	os.chdir(path)
	# print(f'> from: {orig_path}, now currently in: {path}')

	# PARSE XML
	# print('> parsing writers.xml...')
	xml = lxml.etree.parse(xml_path)

	# CREATE CSV FILE
	# print('> creating csv file of forms.txt...')
	forms_txt_to_csv = pd.read_csv(forms_path, delimiter = ' ', usecols = [i for i in range(2)], names=['form_id', 'writer_id', 'C'] )
	forms_txt_to_csv.to_csv('forms.csv', index = None)

	# print('> creating writers.csv from writers.xml...')
	csvfile = open('writers.csv','w',encoding='utf-8',newline='')
	csvfile_writer = csv.writer(csvfile)
	csv_line = ['writer_id', 'gender', 'handedness']
	csvfile_writer.writerow(csv_line)

	# FOR EACH WRITER
	for writer in xml.xpath('//Writer'):

		  # EXTRACT WRITER DETAILS  
		  id = writer.attrib['name']
		  gender = writer.attrib['Gender']
		  handedness = writer.attrib['WritingType']
		  csv_line = [id, gender, handedness]

		  # ADD A NEW ROW TO CSV FILE
		  csvfile_writer.writerow(csv_line)
	csvfile.close()

	# print('> creating writing_info database...')
	conn = sql.connect('THESIS_WRITING_INFO.DB')
	cursor = conn.cursor()
	conn.commit()

	try:
	    cursor.execute('CREATE TABLE forms(form_id TEXT, writer_id INTEGER)')
	    conn.commit()
	except Exception as e:
	    # print('> skipped: ' + str(e) + ' already exists')
	    pass

	try:
	    cursor.execute('CREATE TABLE writers(writer_id INTEGER, gender TEXT, handedness TEXT)')
	    conn.commit()
	except Exception as e:
	    # print('> skipped: ' + str(e) + ' already exists')
	    pass

	# print('> reading forms.csv and writers.csv...')
	forms_csv = pd.read_csv('forms.csv')
	writers_csv = pd.read_csv('writers.csv')

	forms_csv.to_sql(name = 'forms', con = conn, index = False, if_exists = 'replace')
	writers_csv.to_sql(name = 'writers', con = conn, index = False, if_exists = 'replace')

	forms = pd.read_sql_query("SELECT * FROM forms",conn)
	writers = pd.read_sql_query("SELECT * FROM writers",conn)
	writing_info = pd.read_sql_query("SELECT * FROM writers INNER JOIN forms USING(writer_id) ORDER BY writer_id",conn)

	print('> creating writing_info.csv...')
	writing_info.to_csv('writing_info.csv', header=False, index=False)

	os.chdir(orig_path)
	# print(f'> now back in main path: {orig_path}')

def split_data_and_counter(split_data, augmentation=True):
	if not augmentation:
		least = min(split_data['male']['train'], split_data['female']['train'])
		split_data['male']['train'] = least
		split_data['female']['train'] = least

		least = min(split_data['right-handed']['train'], split_data['left-handed']['train'])
		split_data['right-handed']['train'] = least
		split_data['left-handed']['train'] = least

		least = min(
			split_data['right-handed-male']['train'], 
			split_data['left-handed-male']['train'],
			split_data['right-handed-female']['train'], 
			split_data['left-handed-female']['train'],
			)
		split_data['right-handed-male']['train'] = least
		split_data['left-handed-male']['train'] = least
		split_data['right-handed-female']['train'] = least
		split_data['left-handed-female']['train'] = least

	split_counter = {
		'male': {'train': 0, 'test': 0}, 
		'female': {'train': 0, 'test': 0},
		'right-handed': {'train': 0,  'test': 0},  
		'left-handed': {'train': 0,'test': 0},
		'right-handed-male': {'train': 0,  'test': 0},
		'left-handed-male': {'train': 0,  'test': 0},
		'right-handed-female': {'train': 0, 'test': 0},
		'left-handed-female': {'train': 0, 'test': 0}
	}

	return split_data, split_counter

def split_lineImages_random( # Will not work if multiple people used this with the dataset
	split_data, 
	root='data', 
	path='data', 
	metadata_path='data/metadata',
	writer_path='writing_info.csv', 
	category=None, 
	augmentation=True
):
	lineImages = f'{root}/lineImages' #lineImages folder path
	lineSplitDataset = f'{path}/lineSplitDataset' #lineSplitDataset folder path

	print('> creating lineSplitDataset directory...')
	if category:
		make_dirs(lineSplitDataset, one='test', category=category)
		make_dirs(lineSplitDataset, one='train', category=category)
	else:
		for categ, perclass in classes_dict.items():
			make_dirs(lineSplitDataset, one='test', category=categ)
			make_dirs(lineSplitDataset, one='train', category=categ)

	# create dictionary for the info of writers
	writing_info = dict()
	filename = f'{metadata_path}/{writer_path}'

	print('> reading writing_info.csv...')
	#open writing_info.csv
	with open(filename, 'r') as csvfile:
		datareader = csv.reader(csvfile)
		#get each row of csv 
		for row in datareader:
			#add each row to writing_info
			writing_info[row[3]] = {
				'gender': row[1].lower(),
				'handedness' :row[2].lower(),
			}
	
	abs_path = os.getcwd()
	lineImages = os.path.join(abs_path, lineImages)
	lineImages = lineImages.replace('\\', '/')
	lineSplitDataset = os.path.join(abs_path, lineSplitDataset)
	lineSplitDataset = lineSplitDataset.replace('\\', '/')

	split_data, split_counter = split_data_and_counter(split_data, augmentation=augmentation)
	
	# iterate through the file names in files
	# store fpaths with key form id
	print('> splitting lineImages by category and its classes...')

	# if theres json file of this
	fpaths_with_randomsplit = f'{path}/fpaths_with_randomsplit.json'
	if os.path.isfile(fpaths_with_randomsplit):
		fpaths = _load_json(fpaths_with_randomsplit)['fpaths']
		for i in range(len(fpaths)):
			path = fpaths[i].split('/')
			fpaths[i] = f'{os.getcwd()}\\{path[-2]}\\{path[-1]}'
	# else make the file
	else:
		fpaths = list(glob.iglob(lineImages+'/**/*.tif', recursive=True))
		random.shuffle(fpaths)
		data = {'fpaths': fpaths}
		_save_json(f'{path}/fpaths_with_randomsplit.json', data)

	for fpath in fpaths:
		filename = os.path.basename(fpath)
		val = -1
		for i in range(0, 2):
			val = filename.find("-", val + 1)
		form_id = filename[ : val]

		if form_id not in writing_info.keys():
			print(f'could not find form with form id {form_id}')
			continue

		# category
		gender = writing_info[form_id]['gender'].lower()
		handedness = writing_info[form_id]['handedness'].lower()
		gh = handedness + '-' + gender
		perclass_writer = {
			'gender-handedness': gh, # left-handed-male
			'handedness': handedness, # left-handed
			'gender': gender, # male
		}
		for categ, writerclass in perclass_writer.items():
			# train or test
			tt_list = ['test', 'train']
			tt = random.choice(tt_list)
			tt_list.remove(tt)

			# if full, go the other one
			if split_counter[writerclass][tt] >= split_data[writerclass][tt]:
				tt = tt_list[0]

			# if still full, skip
			if split_counter[writerclass][tt] >= split_data[writerclass][tt]:
				continue

			# copy images
			if category:
				if categ == category:
					dst = f'{lineSplitDataset}/{categ}/{tt}/{writerclass}'
				else:
					continue
			else:
				dst = f'{lineSplitDataset}/{categ}/{tt}/{writerclass}'

			shutil.copy2(fpath, dst)
			split_counter[writerclass][tt] += 1

def split_lineImages_stratified(
	split_data,
	root='data', 
	path='data', 
	metadata_path='data/metadata', 
	writer_path='writing_info.csv', 
	category=None,
	augmentation=True
):
	lineImages = f'{root}/lineImages' #lineImages folder path
	lineSplitDataset = f'{path}/lineSplitDataset' #lineSplitDataset folder path

	print('> creating lineSplitDataset directory...')
	if category:
		make_dirs(lineSplitDataset, one='test', category=category)
		make_dirs(lineSplitDataset, one='train', category=category)
	else:
		for categ, perclass in classes_dict.items():
			make_dirs(lineSplitDataset, one='test', category=categ)
			make_dirs(lineSplitDataset, one='train', category=categ)

	# create dictionary for the info of writers
	writing_info_with_writersplit = f'{path}/writing_info_with_writersplit.json'

	print('> reading writing_info.csv...')
	if os.path.isfile(writing_info_with_writersplit):
		print(f'{writing_info_with_writersplit} found. Loading json file.')
		writing_info = _load_json(writing_info_with_writersplit)
	else:
		writing_info = dict()
		filename = f'{metadata_path}/{writer_path}'
		#open writing_info.csv
		with open(filename, 'r') as csvfile:
			datareader = csv.reader(csvfile)
			#get each row of csv 
			for row in datareader:
				#add each row to writing_info
				if row[0] not in writing_info:
					writing_info[row[0]] = {
						'gender': row[1].lower(),
						'handedness' :row[2].lower(),
						'form_ids': [row[3]]
					}
				else:
					writing_info[row[0]]['form_ids'].append(row[3])
					random.shuffle(writing_info[row[0]]['form_ids']) # randomize
	
	abs_path = os.getcwd()
	lineImages = os.path.join(abs_path, lineImages)
	lineImages = lineImages.replace('\\', '/')
	lineSplitDataset = os.path.join(abs_path, lineSplitDataset)
	lineSplitDataset = lineSplitDataset.replace('\\', '/')

	# change split data if no augmentation
	split_data, split_counter = split_data_and_counter(split_data, augmentation=augmentation)
	
	# iterate through the file names in files
	# store fpaths with key form_id
	form_dict_with_writersplit = f'{path}/form_dict_with_writersplit.json'

	print('> splitting lineImages by category and its classes...')
	if os.path.isfile(form_dict_with_writersplit):
		print(f'{form_dict_with_writersplit} found. Loading json file. Parsing path.')
		form_dict = _load_json(form_dict_with_writersplit)
		for form_id, fpaths in form_dict.items():
			for i in range(len(fpaths)):
				fpath = fpaths[i].split('/')
				fpaths[i] = f'{os.getcwd()}\\{fpath[-2]}\\{fpath[-1]}'
	else:
		form_dict = dict()

		fpaths = list(glob.iglob(lineImages+'/**/*.tif', recursive=True))
		random.shuffle(fpaths)

		for fpath in fpaths:
			filename = os.path.basename(fpath)
			val = -1
			for i in range(0, 2):
				val = filename.find("-", val + 1)
			form_id = filename[ : val]

			if form_id not in form_dict:
				form_dict[form_id] = [fpath]
			else:
				form_dict[form_id].append(fpath)

	if not os.path.isfile(writing_info_with_writersplit):
		_save_json(writing_info_with_writersplit, writing_info)

	if not os.path.isfile(form_dict_with_writersplit):
		_save_json(form_dict_with_writersplit, form_dict)

	# for each writer, check first if their line images can fit in split_data
	# then, if amount of images can fit -> training / testing
	missing = []
	included = {
		'gender-handedness': {
			'train': [],
			'test': [],
		},
		'handedness': {
			'train': [],
			'test': [],
		},
		'gender': {
			'train': [],
			'test': [],
		},
	}
	while True:
		delete = []
		for writer_id, val_dict in writing_info.items():

			# load one random fpath of this writer
			form_id = None
			while True:
				if val_dict['form_ids'] == []:
					break

				_id = val_dict['form_ids'][0]
				if _id in form_dict:
					form_id = _id
					break
				else:
					# print(f'missing form: {_id} [{writer_id}]')
					missing.append((_id, writer_id))
					del val_dict['form_ids'][0]
				if val_dict['form_ids'] == []:
					break

			if form_id is None:
				# print('No more forms to process with writer ' + writer_id)
				delete.append(writer_id)
				continue

			fpath = form_dict[form_id].pop(0) # remove from list of fpaths for this form_id
			if form_dict[form_id] == []: # if empty
				del form_dict[form_id] # remove the empty form_id list of fpaths
				del val_dict['form_ids'][0]

			perclass_writer = {
				'gender-handedness': f'{val_dict["handedness"]}-{val_dict["gender"]}', # left-handed-male
				'handedness': val_dict['handedness'], # left-handed
				'gender': val_dict['gender'], # male
			}

			for categ, writerclass in perclass_writer.items():
				# if can fit in test
				if split_counter[writerclass]['test'] < split_data[writerclass]['test']:
					tt = 'test'
				# if valid is full, go train
				elif split_counter[writerclass]['train'] < split_data[writerclass]['train']:
					tt = 'train'
				# if all full, skip
				else:
					continue

				# copy images
				if category:
					if categ == category:
						dst = f'{lineSplitDataset}/{categ}/{tt}/{writerclass}'
					else:
						continue
				else:
					dst = f'{lineSplitDataset}/{categ}/{tt}/{writerclass}'

				print(f'\r[{tt}] {writer_id}: {fpath}', end='')
				if writer_id not in included[categ][tt]:
					included[categ][tt].append(writer_id)
				shutil.copy2(fpath, dst)
				split_counter[writerclass][tt] += 1
				
		for writer_id in delete:
			del writing_info[writer_id]

		if writing_info == {}:
			# print(form_dict)
			break

	# print(included)
	# print(missing)
	included_writersplit = f'{path}/included_writersplit.json'
	if not os.path.isfile(included_writersplit):
		_save_json(included_writersplit, included)

def split_lineImages_main(
	split_data,
	stratified_split=False,
	random_split=False,
	root='data',
	path='data', 
	metadata_path='data/metadata',
	writer_path='writing_info.csv', 
	category=None, 
	augmentation=True
):
	if stratified_split:
		split_lineImages_stratified(split_data, root=root, path=path, category=category, augmentation=augmentation)
	elif random_split:
		split_lineImages_random(split_data, root=root, path=path, category=category, augmentation=augmentation)
	else:
		raise Exception('No choice for split was given -dev')

	return True



### PHASE 2 - segmentation of line images
def word_segmentation_function(img):
    h, w, c = img.shape

    if w > 1000:
        new_w = 1000
        ar = w/h
        new_h = int(new_w/ar)
        img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)

    def thresholding(image):
        img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
        # plt.imshow(thresh, cmap='gray')
        return thresh

    thresh_img = thresholding(img);

    # dilation
    # **************** np.ones((X1, Y1), np.uint8) ****************
    kernel = np.ones((3,25), np.uint8)
    # kernel = np.ones((50,20), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations = 1)

    (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h)

    img2 = img.copy()

    for ctr in sorted_contours_lines:
        x,y,w,h = cv2.boundingRect(ctr)
        cv2.rectangle(img2, (x,y), (x+w, y+h), (40, 100, 250), 2)

    # dilation
    # **************** np.ones((X2, Y2), np.uint8) ****************
    kernel = np.ones((3,20), np.uint8)
    # kernel = np.ones((3,15), np.uint8)
    dilated2 = cv2.dilate(thresh_img, kernel, iterations = 1)

    img3 = img.copy()
    words_list = []

    for line in sorted_contours_lines:
        # roi of each line
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated2[y:y+w, x:x+w]

        # draw contours on each word
        (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contour_words = sorted(cnt, key=lambda cntr : cv2.boundingRect(cntr)[0])

        for word in sorted_contour_words:
        	# **************** < Z ****************
            if cv2.contourArea(word) < 1200:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(word)
            words_list.append([x+x2, y+y2, x+x2+w2, y+y2+h2])
            cv2.rectangle(img3, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (255,255,100),2)

    for on in range(len(words_list)):
        n_word = words_list[on]
        roi = img[n_word[1]:n_word[3], n_word[0]:n_word[2]]

        yield roi

def segmentation_process(src_path=None):
	print(src_path)
	lines = [] # segmented
	total_words = 0
	for i, fpath in enumerate(glob.glob(f'{src_path}/*.tif')):
		print(f'{i+1} line images segmented', end='\r')
		words = []
		words += word_segmentation_function(cv2.imread(fpath))
		total_words += len(words)
		lines.append(words)
	print()
	print(f'{total_words} total words after segmentation')
	return lines



### PHASE 2 - augmentation of trainvalid, testing as is
# lineSplitDataset => wordImages

def image_rotation(w):
	# choose random inclination angle
	# i = random.choice([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10])
	i = random.choice([-10, -5, 0, 5, 10])
	# rotate the image
	w_r = imutils.rotate_bound(w, i)
	# make a white box with same dim as image and rotate the same way
	(height, width) = w.shape[:2]
	img_1 = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8)
	img_1 = imutils.rotate_bound(img_1, i)
	# make a mask with the rotated white box to fill the black gaps of the original image
	mask = img_1 > 250
	selection = w_r.copy()
	selection[~mask] = 255
	w_r = selection
	# return the image and angle used
	return w_r, i

def positive_scaling(w_r):
    (height, width) = w_r.shape[:2]
    # choose random scaling percentages
    # vs = random.choice([0, 0.05, 0.10, 0.15, 0.20])
    # hs = random.choice([0, 0.05, 0.10, 0.15, 0.20])
    vs = random.choice([0, 0.10, 0.20])
    hs = random.choice([0, 0.10, 0.20])
    # rescale the image
    rescaled = cv2.resize(w_r,(int(width + width*hs), int(height + height*vs)), interpolation = cv2.INTER_LINEAR)
    rescaled = Image.fromarray(rescaled)
    # make a padded canvas to not crop the rescaled image
    pad_h, pad_w = int(height*0.15), int(width*0.15)
    new_h, new_w = height + pad_h, width + pad_w
    mgn_h, mgn_w = int(pad_h - vs * height) // 2, int(pad_w - hs * width) // 2
    canvas = Image.new(rescaled.mode, (new_w, new_h), (255, 255, 255))
    canvas.paste(rescaled, (mgn_w, mgn_h))
    w_rs = np.array(canvas)
    # return the image and scaling used
    return w_rs, (vs, hs)

def morph_filter(w_rs):
    # binary morphological filter with a 3x3 structuring element
    m = random.choice(['erosion', 'dilation', None])
    k = None
    if m is not None:
        # k = random.choice([2, 3])
        k = 3
        kernel = np.ones((k, k), np.uint8)
        w_rsm = cv2.dilate(w_rs, kernel, iterations=1) if m == 'erosion' \
        else cv2.erode(w_rs, kernel, iterations=1)
    return (w_rsm if m else w_rs), (m, k)

def padded_resize(image, img_size):
	image = image.astype('uint8')
	image = 255 - image # invert color

	# code from Keras (modified)
	h, w = img_size 
	image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

	# Check the amount of padding needed to be done.
	pad_height = h - tf.shape(image)[0]
	pad_width = w - tf.shape(image)[1]

	# Only necessary if you want to do same amount of padding on both sides.
	if pad_height % 2 != 0:
		height = pad_height // 2
		pad_height_top = height + 1
		pad_height_bottom = height
	else:
		pad_height_top = pad_height_bottom = pad_height // 2

	if pad_width % 2 != 0:
		width = pad_width // 2
		pad_width_left = width + 1
		pad_width_right = width
	else:
		pad_width_left = pad_width_right = pad_width // 2

	image = tf.pad(
		image,
		paddings=[
			[pad_height_top, pad_height_bottom],
			[pad_width_left, pad_width_right],
			[0, 0],
		],
	)

	image = image.numpy().astype('uint8')
	image = 255 - image # invert color

	return image

def resize_input(w_rsm):
	# produce rescaled image w_o using bilinear interpolation to resize
	w_o = cv2.resize(w_rsm, (IMAGE_SIZE['width'], IMAGE_SIZE['height']), interpolation = cv2.INTER_LINEAR) # input training image for the CNN
	return w_o

def augmentation_processes(w):
	w_r, degrees = image_rotation(w)
	w_rs, scale_vh = positive_scaling(w_r)
	w_rsm, morph_kern = morph_filter(w_rs)
	w_o = padded_resize(w_rsm, (IMAGE_SIZE['height'], IMAGE_SIZE['width']))
	# w_o = resize_input(w_rsm)
	return w_o, (degrees, scale_vh, morph_kern)

def augmentation_trainvalid(word_images=None, save_path=None, count=None, ext='tif', aug_combinations=None):
	aug_counter = 0
	while aug_counter < count:
		for idx, image in enumerate(word_images):
			if aug_combinations is None:
				img, combi = augmentation_processes(image)
			else:
				while True:
					img, combi = augmentation_processes(image)
					if idx not in aug_combinations.keys():
						aug_combinations[idx] = [combi]
						break
					elif combi not in aug_combinations[idx]:
						aug_combinations[idx].append(combi)
						break

			cv2.imwrite(f'{save_path}/{aug_counter}.{ext}', img)

			aug_counter += 1
			print(f'{aug_counter} images augmented', end='\r')
			if aug_counter == count:
				break
	print()
	return aug_combinations

def resize_word_images(word_images=None, save_path=None, ext='tif'):
	resized_counter = 0
	for image in word_images:
		img = padded_resize(image, (IMAGE_SIZE['height'], IMAGE_SIZE['width']))
		# img = resize_input(image)

		cv2.imwrite(f'{save_path}/{resized_counter}.{ext}', img)

		resized_counter += 1
		print(f'{resized_counter} images resized', end='\r')
	print()

def word_images(path='data', category=None, augmentation=True):
	lineSplitDataset = f'{path}/lineSplitDataset'
	wordImages = f'{path}/wordImages'

	# make directories
	print('> creating wordImages directory...')
	if category:
		make_dirs(wordImages, one='training', category=category)
		make_dirs(wordImages, one='validation', category=category)
	else:
		if not make_dirs(wordImages, train_valid=True):
			return

	#iterate through the file names in files
	abs_path = os.getcwd()
	wordImages = os.path.join(abs_path, wordImages)
	wordImages = wordImages.replace('\\', '/')
	lineSplitDataset = os.path.join(abs_path, lineSplitDataset)
	lineSplitDataset = lineSplitDataset.replace('\\', '/')

	# split of training and validation dataset according to morera's distribution
	split_tv_dict = { 
		'gender': {'training':100e3 // 2, 'validation':20e3 // 2},
		'handedness': {'training':100e3 // 2, 'validation':25e3 // 2},
		'gender-handedness': {'training':130e3 // 4, 'validation':20e3 // 4}
	}
	
	categ = category
	for category, classes in classes_dict.items():
		if categ and categ != category: 
			continue

		if augmentation:
			for perclass in classes:
				line_images = segmentation_process(f'{lineSplitDataset}/{category}/train/{perclass}')
				word_images = []
				for lines in line_images:
					word_images += lines

				aug_combinations = {}
				print(f'> augmenting training dataset word images: [{category}] [{perclass}]...')
				aug_combinations = augmentation_trainvalid(
					word_images=word_images.copy(),
					save_path=f'{wordImages}/{category}/training/{perclass}',
					count=split_tv_dict[category]['training'],
					aug_combinations=aug_combinations,
				)

				print(f'> augmenting validation dataset word images: [{category}] [{perclass}]...')
				aug_combinations = augmentation_trainvalid(
					word_images=word_images.copy(),
					save_path=f'{wordImages}/{category}/validation/{perclass}',
					count=split_tv_dict[category]['validation'],
					aug_combinations=aug_combinations,
				)
		else:
			cap = {
				'count': 100000,
				'class': None,
			}
			line_images = {}
			for perclass in classes:
				line_images[perclass] = segmentation_process(src_path=f'{lineSplitDataset}/{category}/train/{perclass}')
				word_count = 0
				for lines in line_images[perclass]:
					word_count += len(lines)
				if word_count < cap['count']:
					cap['count'] = word_count
					cap['class'] = perclass

			for perclass in classes:
				word_images = []
				while len(word_images) < cap['count']:
					for lines in line_images[perclass]:
						if lines != []:
							word_images.append(lines.pop(0))
						if len(word_images) == cap['count']:
							break

				print(f'> resizing training dataset word images: [{category}] [{perclass}]...')
				resize_word_images(
					word_images=word_images, 
					save_path=f'{wordImages}/{category}/training/{perclass}',
				)
	return True


def testRunError(m=None):
	raise Exception(
		'huh\n-dev' if not m else m)

def printdone(phase, done):
	if done:
		if phase == 1:
			print('PHASE 1 DONE: lineImages split by class and traintest. Produced lineSplitDataset')
		elif phase == 2:
			print('PHASE 2 DONE: lineSplitDataset segmented and augmented for training and validation. Produced wordImages')
	else:
		print('PHASE '+str(phase)+' skipped. Folder already exists')
	print('----------------------------------------\n')

############ MAIN ############
def data_preparation(
	root='data',
	path='data',
	startphase=1, 
	endphase=3, 
	category=None, 
	augmentation=True, 
	random_split=False, 
	stratified_split=False,
	dummy=False,
):
	# number of phases
	numphase = 2

	# check space of storage
	space_needed = 15 # gigabytes
	total, used, free = shutil.disk_usage("/")
	free_space_string, free_space_value, free_space_unit = format_size(free)
	if free_space_unit != 'G' or (free_space_unit == 'G' and free_space_value < space_needed):
		print('Plausibe insufficient storage space ({free_space_string} available)')
		return

	# just display and check if correct start and end phase values
	print(f'***************\nSTART PHASE:\t{startphase}\nEND PHASE:\t{endphase}\n***************\n')
	if startphase > numphase or endphase > numphase or startphase < 1 or endphase < 1 or startphase > endphase:
		raise Exception(f'Incorrect startphase/endphase value. There are only {numphase} phases, and no negative values are allowed.\n-dev')

	if startphase == 1:
		# ====== Phase 1 ======
		# lineImages => lineSplitByClass
		make_writing_info() # make writing_info.csv using writers.xml and forms.csv

		# using the splitting morera used (approximately) on the line images
		split_data = {
			'male': {'train': 5228,'test': 1659},
			'female': {'train': 3234,'test': 805},
			'right-handed': {'train': 6946,'test': 3011},
			'left-handed': {'train': 728,'test': 250},
			'right-handed-male': {'train': 4851,'test': 1473},
			'left-handed-male': {'train': 429, 'test': 134},
			'right-handed-female': {'train': 2497, 'test': 1113},
			'left-handed-female': {'train': 260, 'test': 115},
		}
		# original of 297 train and 132 test
		# ratio is 297/132 = 2.25
		# left-handed-female values are changed 
		# because of missing writer info, 297 to 260,
		# and 132 to 115, for train and test respectively
		# ratio is 260/115 ~= 2.26

		if dummy: 
			split_data = {
				'male': {'train': 10,'test': 10},
				'female': {'train': 10,'test': 10},
				'right-handed': {'train': 10,'test': 10},
				'left-handed': {'train': 10,'test': 10},
				'right-handed-male': {'train': 10,'test': 10},
				'left-handed-male': {'train': 10, 'test': 10},
				'right-handed-female': {'train': 10, 'test': 10},
				'left-handed-female': {'train': 10, 'test': 10},
			}

		printdone(1, split_lineImages_main( 
			split_data,
			random_split=random_split, 
			stratified_split=stratified_split,
			root=root,
			path=path,
			category=category, 
			augmentation=augmentation)
		)

	if startphase <= 2 and endphase == 2:
		# ====== Phase 2 ======
		# lineSplitDataset => wordImages
		printdone(2, word_images(
			path=path, 
			category=category, 
			augmentation=augmentation)
		)
		

