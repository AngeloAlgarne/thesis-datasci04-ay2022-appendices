from IPython.display import clear_output
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# dictionary of classes according to their category
CLASSES = {
	'gender': ['male', 'female'],
	'handedness': ['left-handed', 'right-handed'],
	'gender-handedness': ['left-handed-male', 'left-handed-female', 'right-handed-male', 'right-handed-female']
}

# dimension of input images
IMAGE_SIZE = {
	'width': 100,
	'height': 30
}

# common utility functions:
def _load_json(filepath):
	with open(f'{filepath}', 'r') as openfile:
		return json.load(openfile)


def _save_json(filepath, data):
	with open(filepath, 'w') as outfile:
		json.dump(data, outfile, indent=4)


# model utility functions
def get_project_name(nlayer, canny, category):
	return f'{category}_{nlayer}LCNN_{"canny" if canny else "nocanny"}'


def model_metadata(
	path=None, 
	nlayer=None,
	canny=None, 
	category=None, 
	morera=None, 
	project_name=None
):
	if project_name is None:
		project_name = get_project_name(nlayer, canny, category)

	project_path = f'{path}/{project_name}_metadata.json'

	if os.path.isfile(project_path):
		print('Loading existing json file.')
		return load_model_metadata(project_path)

	if not morera:
		best_hp = 'None'
	else: 
		best_hp = 'Not Applicable'

	metadata = {
		'project_name': project_name, # static values
		'canny': str(canny),
		'nlayer': nlayer,
		'category': category,
		'best_hp': best_hp, # will only change once tuning is done
		'max_epoch': -1, # will only change once the training is done
	}

	print('Creating new json file.')

	save_model_metadata(
		project_path=project_path, 
		metadata=metadata,
	)

	return metadata


def save_model_metadata(
	project_path, 
	metadata, 
	update=False
):
	if not update:
		if os.path.isfile(project_path):
			raise Exception('Cannot overwrite existing metadata file for this project.\n-dev')
			return
	_save_json(project_path, metadata)


def load_model_metadata(project_path):
	try:
		return _load_json(project_path)
	except FileNotFoundError:
		raise Exception(f'Metadata for project [{project_name}] could not be found.\n-dev')


def update_model_status(
	project_path, 
	max_epoch=None, 
	best_hp=None
):
	if not (max_epoch or best_hp): 
		raise Exception('What\'s being updated chief? Need max_epoch or best_hp.\n-dev')

	jmd = load_model_metadata(project_path)

	if max_epoch:
		jmd['max_epoch'] = max_epoch
	elif best_hp:
		jmd['best_hp'] = best_hp

	save_model_metadata(project_path, jmd, update=True)


def update_model_retrained(
	project_path, 
	new_max_epoch,
):

	jmd = load_model_metadata(project_path)
	jmd['new_max_epoch'] = new_max_epoch

	save_model_metadata(project_path, jmd, update=True)



# user machine utility functions
def init_user(user_id):
	if not os.path.isfile(f'_init_user_{user_id}.json'):
		print('Initializing User')
		try:
			dist_json = _load_json('_machine_distribution.json')
			models = dist_json[user_id]

			new_json = {
				"user": user_id,
				"assigned": models,
				"finished": [],
				"count": f"0 / {len(models)}"
			}

			_save_json(f'_init_user_{user_id}.json', new_json)
		except FileNotFoundError:
			raise Exception('File missing: _machine_distribution.json\n-dev')
		except KeyError:
			raise Exception('User missing: cannot find user in _machine_distribution.json. Error called from "init_user"\n-dev')


def _parse_model_id(
	model_id,
	return_config=True
):
	model = model_id.split('-')
	if model[0] == 'G':
		category = 'gender'
	elif model[0] == 'H':
		category = 'handedness'
	else:
		category = 'gender-handedness'
	nlayer = int(model[1])
	canny = model[2] == 'WC'
	project_name = get_project_name(nlayer=nlayer, canny=canny, category=category)

	return {
		'category': category,
		'nlayer': nlayer,
		'canny': canny,
		'project_name': project_name,
	}

def _get_manual_config(
	_manual_model_id, 
	dist_json=None, 
	parse_only=False,
):
	if parse_only:
		return _manual_model_id, _parse_model_id(_manual_model_id)

	for user, models in dist_json.items():
		for model_id in models:
			if _manual_model_id == model_id:
				return model_id, _parse_model_id(model_id)

	return None, None

def get_user_config(user_id, _manual_model_id=None):
	try: 
		dist_json = _load_json('_machine_distribution.json')

		if _manual_model_id:
			_manual_model_id = _manual_model_id.split(' ')

		if dist_json[user_id] == [] or (_manual_model_id and _manual_model_id[0] == 'ALLOWED'):
			message = 'You need to enter a model ID'

			if len(_manual_model_id) > 1:
				model_id, model_config = _get_manual_config(_manual_model_id[1], dist_json)

				if model_id is None:
					message = 'You have entered an invalid model_id.'
				else:
					return model_id, model_config

			while True:
				extra = 'model_id\n********\nG-2-NC\nH-2-NC\nGH-2-NC\nG-3-NC\nH-3-NC'
				extra+= '\nG-2-WC\nH-2-WC\nGH-2-WC\nG-3-WC\nH-3-WC\nGH-3-WC\n********'
				print(extra)
				extra2 = f'\nHi! {message}. Ask Angelo if you dont know! :)'
				_manual_model_id = input(f'{extra2}\nModel ID (model_id): ')
				model_id, model_config = _get_manual_config(_manual_model_id, dist_json)

				if model_id is None:
					clear_output()
					message = 'You have entered an invalid model_id.'
				else:
					return model_id, model_config
		else:
			user_json = _load_json(f'_init_user_{user_id}.json')
			for model_id in dist_json[user_id]:
				if model_id in user_json['finished']:
					continue
				else:
					return _get_manual_config(model_id, parse_only=True)
	except FileNotFoundError:
		raise Exception(f'File missing: _init_user_{user_id}.json / _machine_distribution.json\n-dev')
	except KeyError:
		raise Exception('User missing: cannot find user in _machine_distribution.json. Error called from "get_user_config"\n-dev')
	return None, None

def update_user(user_id, model_id):
	try:
		user_json = _load_json(f'_init_user_{user_id}.json')
		user_json['finished'].append(model_id)
		user_json['count'] = f'{len(user_json["finished"])} / {len(user_json["assigned"])}'
		
		_save_json(f'_init_user_{user_id}.json', user_json)

		# dist_json = _load_json('_machine_distribution.json')
		# if user_json['finished'] == dist_json[user_json['user']]:
		# 	return False
		# return True
	except FileNotFoundError:
		raise Exception(f'File missing: _init_user_{user_id}.json\n-dev')

def is_user_done(user_id):
	try:
		user_json = _load_json(f'_init_user_{user_id}.json')

		assigned = user_json['assigned']
		finished = user_json['finished']

		for assigned_model in assigned:
			if assigned_model not in finished:
				return False
		return True
	except FileNotFoundError:
		raise Exception(f'File missing: _init_user_{user_id}.json\n-dev')



# graphing utility functions
def _get_preprocessed_df(
	path, 
	columns
):
	df = pd.read_csv(path)
	return df[columns]


def plot_trials(
	tuner_output_path, 
	columns=['accuracy', 'val_accuracy'], 
	trial_count=10, 
	column_count = 5, 
	row_count = 2
):
	trial_df = []
	trial_lbl = []
	for i in range(trial_count):
		trial_lbl.append(f'trial_0{i}')
		trial_df.append(
			_get_preprocessed_df(
				f'{tuner_output_path}/trial_0{i}.csv',
				columns,
			)
		)

	fig, ax = plt.subplots(row_count, column_count, figsize=(15,6), constrained_layout=True)
	fig.tight_layout()

	fig.suptitle(tuner_output_path.split('/')[-1][:-13])

	y = -1
	for i in range(10):
		x = i % column_count
		if x == 0:
			y += 1
		axis = ax[y, x]
		sns.lineplot(data=trial_df[i], ax=axis)
		axis.set_title(trial_lbl[i])

	plt.subplots_adjust(bottom=0.01, top=0.9)
	plt.show()


def plot_model(
	log_path, 
	columns=['accuracy', 'val_accuracy'],
):
	if type(log_path) is not list:
		plt.figure(figsize=(5,5))
		plt.title(path.split('/')[-1][:-10])
		sns.lineplot(
			data=_get_preprocessed_df(
				log_path, 
				columns,
			),
		)
		plt.ylabel('percent')
		plt.xlabel('epochs')
		
		plt.show()
	else:
		trial_df = []
		trial_lbl = []
		for path in log_path:
			trial_lbl.append(path.split('/')[-1][:-10])
			trial_df.append(_get_preprocessed_df(path,columns))
		fig, ax = plt.subplots(1, len(log_path), figsize=(15,5))
		fig.tight_layout()

		for i in range(len(log_path)):
			axis = ax[i]
			sns.lineplot(data=trial_df[i], ax=axis)
			axis.set_title(trial_lbl[i])
			axis.set_ylabel('percent')
			axis.set_xlabel('epochs')
			
		plt.show()


def plot_performance_metrics(
	project_name,
	save_path=None,
	json_path=None,
):
	if save_path:
		project_dir = f'{save_path}/{project_name}'
		if not os.path.isdir(project_dir):
			raise Exception(f'Project directory missing: {project_dir} -dev')

		results = _load_json(f'{project_dir}/{project_name}_results.json')
		if not results:
			raise Exception('No results yet for this project -dev')
		classification_metrics = results['classification_metrics']

	elif json_path:
		results = _load_json(json_path)
		if not results:
			raise Exception('No json found with this path -dev')
		classification_metrics = results['classification_metrics']
	else:
		raise Exception('Need to pass argument either save_path or json_path to "plot_performance_metrics" -dev')

	show_classification_metrics(
		classification_metrics, 
		project_name,
		project_name.split('_')[0],
	)


def show_classification_metrics(
	classification_metrics,
	project_name,
	category,
):
	class_names = CLASSES[category]

	print(f'\nWith the model [{project_name}],', end=' ')
	print('the performance metrics using the testing dataset are the following.\n')

	# display accuracy
	print('accuracy: ', end=' ')
	print(classification_metrics['accuracy'], end='\n\n')

	# display classification report/summary
	print(classification_metrics['report'])

	plt.xticks(
	    rotation=45, 
	    horizontalalignment='right',
	)

	# display confusion matrix
	conf_matrix = np.asarray(classification_metrics['matrix'])
	ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
	xtra_lbl = ('('+ project_name + ')' if project_name else '')
	ax.set_title(f'Confusion Matrix {xtra_lbl}\n')
	ax.set_xlabel('\nPredicted Values')
	ax.set_ylabel('Actual Values')
	ax.tick_params(axis='y', labelrotation = 45)
	ax.xaxis.set_ticklabels(class_names)
	ax.yaxis.set_ticklabels(class_names)

