from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, Callback
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from modules.m2_preprocessing import load_train_valid, load_testing, load_train_valid_tuner, load_testlines
from modules.utils import CLASSES, IMAGE_SIZE, model_metadata, save_model_metadata, \
	update_model_retrained, update_model_status, get_project_name, _load_json, _save_json, show_classification_metrics

from collections import Counter
from scipy.stats import shapiro
from numpy import random

import keras_tuner, json, os, codecs
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import seaborn as sb
import pandas as pd
import scipy.stats as stats



# dictionary for epochs
EPOCHS = {
	'max_dummy': 3,
	'patience_dummy': 3,
	'max': 500,
	'patience': 20, 
	# 'patience': 10, 
	# observation: 15 epochs is the max interval, 
	# gender_2L_nocanny stops learning
}

# dictionary for tuner constants
TUNER = {
	'epochs_dummy': 3,
	'patience_dummy': 3,
	'max_trials_dummy': 3,
	'epochs': 200,
	'patience': 10,
	'max_trials': 10,
	'objective': 'val_accuracy',
	'overwrite': False,
}

class TrainingProcess():
	def __init__(
		self, 
		save_path, 
		dataset_path, 
		nlayer,
		canny,
		category,
		dummy=False,
	):

		# --------- model metadata --------- 
		self.dataset_path = dataset_path
		self.project_name = get_project_name(nlayer, canny, category)
		self.save_path = self.init_save_path(save_path)
		self.nlayer = nlayer
		self.canny = canny
		self.category = category
		self.binary = category != 'gender-handedness'
		self.morera = not canny and nlayer == 2
		metadata = model_metadata( # from utils.py
			path=self.save_path,
			nlayer=nlayer,
			canny=canny, 
			category=category,
			morera=self.morera)

		# --------- if already done ---------
		if metadata['max_epoch'] > -1:
			raise Exception('Training process for this model is already completed. If you want to redo, delete or rename the project folder first.\n-dev')

		# ----------- set paths ------------
		rootprefix = f'{self.save_path}/{self.project_name}'
		self.metadata = rootprefix + '_metadata.json'
		self.checkpointfile = rootprefix + '_model.h5'
		self.csvloggerfile = rootprefix + '_model.csv'
		self.historyfile = rootprefix + '_history.json'

		# -------- if dummy test run --------
		self.dummy = dummy
		self.dummy_suffix = '_dummy' if self.dummy else ''
		
		# ------- set number of epochs ------
		self.epochs = EPOCHS['max'+self.dummy_suffix]
		self.patience = EPOCHS['patience'+self.dummy_suffix]

		# ------ if not morera's archi ------
		if not self.morera:
			self.save_path_tuner = self.init_save_path_tuner(self.save_path)
			self.checkpointtuner = self.save_path_tuner + '/latest_trial_model.h5'
			self.istuned = metadata['best_hp'] != 'None'
			self.epochs_tuner = TUNER['epochs'+self.dummy_suffix]
			self.patience_tuner = TUNER['patience'+self.dummy_suffix]
			self.tunername = self.project_name + '_tuner'

		# ----------- load model ------------ 
		self.model = self.load_model()  # from this class

		# -------- done initializing ----------
		print(f'\nMODEL: {self.project_name}\n(project_name)\n\nInitialized!')


	# creates and returns a save path for the project
	# if already a dir, it just returns the path
	def init_save_path(
		self, 
		save_path
	):
		save_path = save_path + '/' + self.project_name

		if not os.path.isdir(save_path):
			os.makedirs(save_path)

		return save_path


	# creates and returns a save path for the tuner
	# if already a dir, it just returns the path
	def init_save_path_tuner(
		self, 
		save_path
	):
		# save_path is already output/project_name
		save_path = save_path + '/' + self.project_name + '_tuner_output'

		# save_path will be output/project_name/project_name_tuner_output
		if not os.path.isdir(save_path):
			os.makedirs(save_path)

		return save_path


	# returns the number of trials
	# raises exception if didnt find anything
	def get_current_trials(self):
		max_trials = TUNER['max_trials'+self.dummy_suffix]
		trials = 0
		for trials in range(max_trials):
			trial_dir = 'trial_'
			if trials < 10 and max_trials >= 10:
				trial_dir += '0'

			if not os.path.isdir(f'{self.save_path}/{self.tunername}/{trial_dir}{trials}'):
				return trials
		return max_trials
		raise Exception('Bug. Did not return anything from function get_current_trials. -dev')


	# returns the number of epochs
	# returns 0 if no csv logger
	def get_max_epoch(
		self, 
		csvlogger,
	):
		try: # get max epoch from the logger file path
			logs = pd.read_csv(csvlogger) # get logs
			return logs.shape[0]
		# if log not found, must be new
		except FileNotFoundError:
			return 0


	# if existing, returns the h5 file model
	# else, returns the base model, compiled if morera's
	def load_model(self):
		if os.path.isfile(self.checkpointfile):
			return keras_load_model(self.checkpointfile, compile=True) # from keras
		else:
			if self.morera:
				return self.build_model(None)
			else:
				csvloggerfile_tuner = f'{self.save_path_tuner}/latest_trial_log.csv'
				if os.path.isfile(csvloggerfile_tuner) and os.path.isfile(self.checkpointtuner):
					return keras_load_model(self.checkpointtuner, compile=True) # from keras
				return None


	# uses the entire function and compiles the base model if morera's
	# takes the entire function and is passed to the tuner, if tuner
	def build_model(
		self, 
		hp,
	):
		input_shape = (IMAGE_SIZE['height'], IMAGE_SIZE['width'], 1)

		# ========== morera's ==========
		learning_rate = 0.001
		kernel_size = (5,5)
		maxpooling_size = (2,2)
		padding = 'same' # zero-padding
		dropout_layer = 0.25
		dropout_dense = 0.5
		units = 512 # last dense layer
		weight_decay = 1E-7

		# constant
		activation_layer = 'relu'
		activation_dense = 'softmax'

		if self.category == 'gender':
			filters = 128
		elif self.category == 'handedness':
			filters = 64
		else:
			filters = 32

		# ========== start of bayesian optimization ==========
		if hp:
			filters = hp.Int('filters', min_value=32, max_value=128, step=32)

			learning_rate = hp.Float(
				'learning_rate', 
				min_value=1e-4, 
				max_value=1e-2,
				sampling='log',
			)

			dropout_layer = hp.Float(
				'dropout_layer', 
				min_value=0.0001, 
				max_value=0.5, 
				sampling='log', 
			)

			dropout_dense = hp.Float(
				'dropout_dense', 
				min_value=0.0001, 
				max_value=0.5, 
				sampling='log', 
			)

			weight_decay = hp.Float(
				'weight_decay', 
				min_value=1e-8, 
				max_value=1e-6,
				sampling='log',
			)

		# Model
		model = Sequential()

		# First block
		model.add(Conv2D(filters, # 32
			kernel_size, 
			input_shape=input_shape, 
			padding=padding,
		))
		model.add(Activation(activation_layer)) # relu: activates the input of block to block
		model.add(Dropout(dropout_layer)) # reduces number of neurons randomly
		model.add(MaxPooling2D(maxpooling_size)) # summarizes the pixel values which makes the conv smaller essentially

		# Second block
		model.add(Conv2D(filters * 2, # 64
			kernel_size, 
			padding=padding,
		))
		model.add(Activation(activation_layer))
		model.add(Dropout(dropout_layer))
		model.add(MaxPooling2D(maxpooling_size))

		# Third block
		# filters *2 is somewhat a default / norm in setting filters
		if self.nlayer == 3:
			model.add(Conv2D(filters * 4, # 128
				kernel_size, 
				padding=padding,
			))
			model.add(Activation(activation_layer))
			model.add(Dropout(dropout_layer))
			model.add(MaxPooling2D(maxpooling_size))

		# Fully connected blocks
		model.add(Flatten()) # into 1 dimension for dense layer; prep for output
		
		# Output Layer
		model.add(Dense(units)) # summarizes the flattened
		model.add(Activation(activation_layer))
		model.add(Dropout(dropout_dense))
		model.add(Dense(2 if self.binary else 4)) # actual output
		model.add(Activation(activation_dense)) # softmax: activates the input of blocks to prediction

		# Instantiate a optimizer
		if self.binary:
			optimizer = tfa.optimizers.SGDW(
				learning_rate=learning_rate,
				weight_decay=weight_decay)
		else:
			optimizer = tfa.optimizers.AdamW(
				learning_rate=learning_rate,
				weight_decay=weight_decay)


		# Instantiate a logistic loss function that expects integer targets
		loss = 'categorical_crossentropy'
		
		# Instantiate an accuracy metric
		accuracy = 'accuracy'

		model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

		return model


	# callback for search and train
	# makes a folder for logs and tensorboard
	def create_callbacks(
		self, 
		csvlogger, 
		checkpoint, 
		patience,
		monitor,
	):
		return [
			CSVLogger(csvlogger, separator=',', append=True),
			ModelCheckpoint(checkpoint, monitor=monitor, verbose=1, save_best_only=True),
			EarlyStopping(monitor=monitor, patience=patience, verbose=1, baseline=None),
		]


	# search best hyperparameters using bayesian optimization tuner
	# will be skipped if morera, or if tuner is done searching
	def search(
		self, 
		training_dataset_only
	):
		if self.morera: 
			print('\nSearching with Tuner skipped: Morera\'s architecture is being used.')
			return

		# get current number of trials
		trial_count = max(self.get_current_trials(), 1)

		# if theres continuation epoch
		csvloggerfile = f'{self.save_path_tuner}/latest_trial_log.csv'
		cont_epoch = self.get_max_epoch(csvloggerfile)

		# instantiate custom model to tune
		model = MyHyperModel(
			save_path=self.save_path_tuner,
			build_model=self.build_model, 
			category=self.category, 
			nlayer=self.nlayer, 
			binary=self.binary,
			epochs=self.epochs_tuner,
			cont_epoch=cont_epoch,
			model=self.model,
			trial_count=trial_count,
		)

		tuner =  keras_tuner.BayesianOptimization(
			hypermodel=model,
			objective=TUNER['objective'],
			max_trials=TUNER['max_trials'+self.dummy_suffix],
			overwrite=TUNER['overwrite'],
			directory=self.save_path,
			project_name=self.tunername,
		)

		if not self.istuned:
			tuner_train_batch, tuner_valid_batch = load_train_valid_tuner( # from module 2
				path=self.dataset_path, 
				category=self.category, 
				canny=self.canny, 
				dummy=self.dummy,
				training_dataset_only=training_dataset_only,
			)

			tuner.search(
				tuner_train_batch,
				validation_data = tuner_valid_batch,
				verbose = 1,
				callbacks=self.create_callbacks(
					csvlogger=csvloggerfile,
					checkpoint=self.checkpointtuner,
					patience=self.patience_tuner,
					monitor='val_accuracy',
				),
			)

			update_model_status( # from utils.py
				project_path = self.metadata,
				best_hp = tuner.get_best_hyperparameters()[0].values,
			)

		print(f'\nSearching with Tuner finished: {TUNER["max_trials"+self.dummy_suffix]} "max_trial" for the tuner has been reached,')

		# get the top 1 hyperparameters.
		best_hp = tuner.get_best_hyperparameters()[0]

		print('With best hyperparameters:', end=' ')
		print(best_hp.values, end='\n')

		if self.istuned:
			return

		# build the model with the best hp.
		self.model = self.build_model(best_hp)


	# train with best hyperparameters
	# returns the trained model
	def train(
		self, 
		training_dataset_only,
	):

		# get last epoch if stopped, then print message
		cont_epoch = self.get_max_epoch(csvlogger=self.csvloggerfile)
		if cont_epoch > 0:
			self.epochs -= cont_epoch
			message = f'Project stopped training at {cont_epoch} epoch/s... '
			message += f'Continuing with {self.epochs} epoch/s left.'
			print(message, end='\n')

		# load dataset 
		train_batch, valid_batch = load_train_valid( # from module 2
			path=self.dataset_path, 
			category=self.category, 
			canny=self.canny, 
			dummy=self.dummy,
			training_dataset_only=training_dataset_only,
		)

		# start/continue to train the model
		self.model.fit(
			train_batch,
			validation_data=valid_batch, 
			epochs=self.epochs, 
			verbose=1,
			callbacks=self.create_callbacks(
				csvlogger=self.csvloggerfile, 
				checkpoint=self.checkpointfile, 
				patience=self.patience,
				monitor='val_accuracy',
			),
		)

		update_model_status( # from utils.py
			project_path = self.metadata,
			max_epoch = self.get_max_epoch(csvlogger=self.csvloggerfile),
		)


	# function that will start the training process
	# calls search and train function
	def start(
		self, 
		training_dataset_only=False,
	):
		self.search(training_dataset_only)
		self.train(training_dataset_only)



class MyHyperModel(keras_tuner.HyperModel):
	def __init__(
		self,
		save_path,
		build_model, 		# function
		category, 			# metadata
		nlayer, 
		binary,
		epochs,				# epochs of tuner
		cont_epoch,			# continutation epoch
		model,				# h5 file of trial model
		trial_count,		# current number of trials
	):
		super(MyHyperModel, self).__init__()
		self.save_path = save_path
		self.build_model = build_model				
		self.category = category 					
		self.nlayer = nlayer		
		self.binary = binary
		self.epochs = epochs 						
		self.cont_epoch = cont_epoch			
		self.model = model 							
		self.trial_count = trial_count			


	# uses the build of TrainingProcess
	# returns the compiled model
	def build(self, hp):
		model = self.build_model(hp)
		return model


	# custom fit function for this model
	# only used when tuning
	def fit(self, hp, model, *args, **kwargs):
		if self.model is not None:
			model = self.model

		# get last epoch if stopped, then print message
		epochs = self.epochs
		if self.cont_epoch > 0:
			epochs -= self.cont_epoch
			message = f'Project stopped tuning at {self.cont_epoch} epoch/s... '
			message += f'Continuing with {epochs} epoch/s left.'
			print(message, end='\n')

		# fit for one trial
		history = model.fit(
			*args,
			epochs=epochs,
			**kwargs,
		)

		max_trials = TUNER['max_trials']
		trial_suffix = ''
		if self.trial_count < 10 and max_trials >= 10:
			trial_suffix += '0'

		# rename the csv logger to not overwrite it
		os.rename(
			f'{self.save_path}/latest_trial_log.csv', 
			f'{self.save_path}/trial_{trial_suffix}{self.trial_count-1}.csv',

		)

		# increase trial count
		self.trial_count += 1

		# reset cont_epoch, since trial is done
		self.cont_epoch = 0

		return history



class TestingProcess():
	def __init__(self, 
		save_path, 
		dataset_path=None, 
		dummy=False,
		show_skipped=False,
		show_results=False,
		return_skipped=False,
	):
		self.save_path_root = save_path
		self.dataset_path = dataset_path
		self.dummy = dummy
		self.show_skipped = show_skipped
		self.show_results = show_results
		self.return_skipped = return_skipped


	def load_model(
		self,
		checkpointfile,
	):
		if os.path.isfile(checkpointfile):
			return keras_load_model(checkpointfile, compile=True) # from keras
		else:
			return None


	def model_summary(self, project_name):
		project_path = f'{self.save_path_root}/{project_name}'
		model = self.load_model(f'{project_path}/{project_name}_model.h5')
		print(model.summary())


	def test(self, project_name):
		if not self.dataset_path:
			raise Exception('Attempting to test without dataset_path. -dev')

		if type(project_name) is not list:
			project_name = [project_name]

		skipped = {}

		for project in project_name:
			# set path
			project_path = f'{self.save_path_root}/{project}'

			# if project does not exist
			if not os.path.isdir(project_path):
				skipped[project] = 'Project directory not found.'
				continue

			# get metadata
			metadata = model_metadata( # from utils.py
				path=project_path,
				project_name=project,
			)
			canny = metadata['canny'] == 'True'
			category = metadata['category']
			binary = category != 'gender-handedness'

			# if already done with results
			if os.path.isfile(f'{project_path}/{project}_results.json'):	
				skipped[project] = 'Already been tested. Please check the \"results\" json.'
				continue

			# if not done training
			if metadata['max_epoch'] == -1:
				skipped[project] = 'This model has not finished training yet.'
				continue

			# load model and dataset
			model = self.load_model(f'{project_path}/{project}_model.h5')

			# if model is missing
			if model is None:
				skipped[project] = 'The "_model.h5" file is missing.'
				continue

			# prompt project name
			print(f'\nMODEL: {project}\n(project_name)\n\n')

			# dataset path
			if self.dummy:
				dataset_path = 'data/sample'
			else:
				dataset_path = self.dataset_path

			# load the test lines
			test_x, test_y = load_testlines( # from module 2
				path=dataset_path,
				category=category, 
				canny=canny,
				batch_size=1,
				dummy=self.dummy,
			)

			# test the model
			results_data = self.test_category_model(
				model=model,
				test_x=test_x,
				test_y=test_y,
				project_name=project,
				project_path=project_path,
			)

			if self.show_results:
				show_classification_metrics(
					classification_metrics=results_data['classification_metrics'],
					project_name=project,
					category=category,
				)

		if self.show_skipped:
			if skipped != {}:
				print('Skipped:')
				for key, val in skipped.items():
					print(f'{key}: {val}')

		if self.return_skipped:
			return skipped
	

	def majority_voting_scheme(
		self, 
		pred_words_y,
	):
		vote_pred = [] 
		vote_class = []

		for i, y in enumerate(pred_words_y):
			pred_words_y[i] = y[0]

		#get the highest prediction and its class
		for y in pred_words_y:
			pred_conf = max(y)
			pred_class = y.index(pred_conf)
			# print(f'y: {y}, pred_conf: {pred_conf}, pred_class: {pred_class}')
			vote_pred.append(pred_conf)
			vote_class.append(pred_class)

		# convert array into dictionary
		count = Counter(vote_class)

		# traverse dictionary and check majority element
		size = len(vote_class)

		highest = {
			'class': [],
			'votes': 0,
		}
		for (perclass, votes_perclass) in count.items():
			# print(f'class: {perclass}, votes: {votes_perclass}')
			if votes_perclass > highest['votes']: # if found higher
				highest['class'] = [perclass]
				highest['votes'] = votes_perclass
			elif votes_perclass == highest['votes']: # if tie
				highest['class'].append(perclass)
				
		if len(highest['class']) == 1:
			return perclass, {'votes/confidence': f'{highest["votes"]}/{size}'}
		else: # if tie
			highest_conf_pred = 0
			highest_conf_class = 0
			for i in range(len(vote_class)):
				if vote_class[i] in highest['class'] and vote_pred[i] > highest_conf_pred:
					highest_conf_pred = vote_pred[i]
					highest_conf_class = vote_class[i]
			return highest_conf_class, {'votes/confidence': highest_conf_pred}

	def predict(
		self,
		model,
		word_imgs,
	):
		pred_words_y = []
		for word in word_imgs:
			pred_words_y.append(model.predict(word).tolist())
		pred_line, pred_metadata = self.majority_voting_scheme(pred_words_y)
		return pred_line, pred_metadata

	def test_category_model(
		self, 
		model, 
		test_x,
		test_y=None, 
		project_name=None,
		project_path=None,
	):
		line_counter =  0
		pred_y_perword = []
		pred_y = []
		pred_y_md = []
		for data_line, data_label in zip(test_x, test_y):
			pred_words_y = []
			# print(f'LABEL: {data_label}')
			for word in data_line:
				pred_words_y.append(model.predict(word).tolist())
			pred_y_perword.append(pred_words_y)
			pred_line, pred_metadata = self.majority_voting_scheme(pred_words_y)
			pred_y.append(pred_line)
			pred_y_md.append(pred_metadata)
			print(f'{line_counter+1} lines predicted.', end='\r')
			line_counter += 1
		# print()
		print('\nDone predicting\n')
		# print('\npred_y:')
		# print(pred_y)

		# performance metrics
		classification_metrics = self.compute_classification_metrics(
			ground=test_y,
			predicted=pred_y,
		)

		# save all as json
		return self.save_model_results(
			test_y=test_y, 
			pred_y=pred_y,
			pred_y_md=pred_y_md,
			pred_y_perword=pred_y_perword,
			classification_metrics=classification_metrics,
			project_name=project_name,
			path=project_path,
		)


	def compute_classification_metrics(
		self, 
		ground, 
		predicted,
	):
		# accuracy
		acc = accuracy_score(ground, predicted)

		# precision, recall, fscore, support
		p, r, f, s = precision_recall_fscore_support( # from sklearn
			ground, 
			predicted, 
			labels=list(set(ground)),
		)

		# confusion matrix
		conf_matrix = confusion_matrix(ground, predicted)

		# classification report
		report = classification_report(ground, predicted)

		# return metrics
		metrics = {
			'matrix': conf_matrix,
			'accuracy': acc,
			'precision': p,
			'recall': r,
			'fscore': f,
			'support': s,
			'report': report,
		}

		return metrics


	def save_model_results(
		self,
		test_y, 
		pred_y,
		pred_y_md,
		pred_y_perword,
		classification_metrics,
		project_name,
		path,
	):
		# PERFORMANCE METRICS
		matrix = []
		for row in classification_metrics['matrix']:
			matrix.append(row.tolist())
		classification_metrics['matrix'] = matrix
		classification_metrics['precision'] = classification_metrics['precision'].tolist()
		classification_metrics['recall'] = classification_metrics['recall'].tolist()
		classification_metrics['fscore'] = classification_metrics['fscore'].tolist()
		classification_metrics['support'] = classification_metrics['support'].tolist()

		# DATA
		data = {
			'name': project_name,
			'test_y': test_y,
			'pred_y': pred_y,
			'pred_y_md': pred_y_md,
			'pred_y_perword': pred_y_perword,
			'classification_metrics': classification_metrics,
		}

		_save_json(f'{path}/{project_name}_results.json', data)

		return data





