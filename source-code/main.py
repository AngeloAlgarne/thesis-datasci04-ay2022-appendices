'''
------------------ 1 ---------------------
model_ids
********
G-2-NC
H-2-NC
GH-2-NC
G-3-NC
H-3-NC
GH-3-NC
G-2-WC
H-2-WC
GH-2-WC
G-3-WC
H-3-WC
GH-3-WC
********

------------------------------------------
user_ids
********
Gelo
Josh-1
Darius-1
Josh-2
Darius-2
Mamser
********
'''

USER_ID = 'Gelo'

_manual_model_id = None
folder_config = '' # default is '' (single quotes)

dataprep_enabled = False
training_enabled = True
testing_enabled = True

CLEAR_OUTPUT = True
DUMMY = False


'''
   Code below will execute automatically.  
'''

'''
Paste-Lefts by: John Angelo Algarne

(aka Copy-Right? lol?)
'''

import warnings 
warnings.filterwarnings('ignore')

from modules.utils import init_user
init_user(USER_ID)

import json
import os
import traceback
import pandas as pd
from IPython.display import clear_output
import tensorflow.config.experimental as tfce
from modules.m1_datapreparation import data_preparation
from modules.m3_classification import TrainingProcess, TestingProcess
from modules.utils import get_user_config, update_user, is_user_done, \
    _load_json, _parse_model_id
%load_ext jupyternotify
clear_output()

# KEEP THE VALUE TO !! [ T R U E ] !! ==> DO NOT CHANGE
try: tfce.set_memory_growth(tfce.list_physical_devices('GPU')[0], True)
except Exception: pass


def finished():
    %notify -m "Training Process: FINISHED"
    val = 'https://drive.google.com/drive/folders/1WO8hAdyP9NfoGzPX1qmGscFSVWN3h0Lg?usp=sharing'
    df = pd.DataFrame({'link':[val]})
    df.style.format({'url': lambda val: f'<a target="_blank" href="{val}">{val}</a>'})
    print('Hi!\n\nIf there are "Skipped" items above, do not proceed yet and ask Angelo.')
    print('Else, please upload the zipped "output" folder in the link below:')
    print("\n".join(df["link"]))
    print('\nFolder format: <name>_output (e.g., gelo_output)')
    print('\nIn case something happens, please do not delete the files yet.')
    print(f'\nThanks {user}! -dev')  
    if DUMMY:
        with open(f'_init_user_{USER_ID}.json', 'r') as openfile:
            user_json = json.load(openfile)
        user_json['finished'] = []
        user_json['count'] = f'0 / {len(user_json["assigned"])}'
        with open(f'_init_user_{USER_ID}.json', 'w') as outfile:
            json.dump(user_json, outfile, indent=4)
    

# ================================================================================



# PATH = 'C:\\Users\\geloa\\jupyter\\thesis\\data\\orig\\wordImages'
if not DUMMY: 
    PATH = f'data{folder_config}'
    SAVE = f'output{folder_config}'
    TEST = f'data{folder_config}/lineSplitDataset'
    AUGMENTATION = True
else: 
    PATH = 'data/_dummy'
    SAVE = '_dummy' 
    TEST = 'data/_sample'
    AUGMENTATION = False
    
try:
    # ================ DATA PREP ===============
    lineSplitDataset_path = f'{PATH}/lineSplitDataset'
    wordImages_path = f'{PATH}/wordImages'
    
    # ================ TRAINING ================
    while training_enabled:
        model_id, model_config = get_user_config(
            user_id=USER_ID,
            _manual_model_id=_manual_model_id
        )

        if model_id is None:
            break

        N_LAYER = int(model_config['nlayer'])
        CANNY = model_config['canny']
        CATEGORY = model_config['category']
        
        # ****** data prep ******
        if not os.path.isdir(f'{lineSplitDataset_path}/{CATEGORY}'):
            STARTPHASE = 1
        elif not os.path.isdir(f'{wordImages_path}/{CATEGORY}'):
            STARTPHASE = 2
        else:
            STARTPHASE = None      
        if dataprep_enabled:
            if STARTPHASE is not None:
                data_preparation(
                    root='data',
                    path=PATH,
                    startphase=STARTPHASE, 
                    endphase=2,
                    augmentation=AUGMENTATION, # True
                    stratified_split=True, # stratified_split
                    dummy=DUMMY,
                    category=CATEGORY,
                )
                if CLEAR_OUTPUT:
                    clear_output()
        else:
            print('WARNING! dataprep is disabled')

        # ****** training ******
        training_process = TrainingProcess(
            save_path=SAVE,
            dataset_path=wordImages_path,
            nlayer=N_LAYER, 
            canny=CANNY, 
            category=CATEGORY,
            dummy=DUMMY,
        )

        training_process.start(training_dataset_only=DUMMY)
        
        update_user(
            user_id=USER_ID, 
            model_id=model_id
        )

        if CLEAR_OUTPUT:
            clear_output()
        if _manual_model_id or is_user_done(USER_ID):
            break
    if not training_enabled:
        print('WARNING! training is disabled')

    # ================ TESTING ================
    if testing_enabled:
        # all finished models to list
        finished_models = _load_json(f'_init_user_{USER_ID}.json')['finished']
        models_project_name = []
        for fm in finished_models:
            parsed = _parse_model_id(fm)
            models_project_name.append(parsed['project_name'])

        # initialize testing
        testing_process = TestingProcess(
            save_path=SAVE,
            dataset_path=TEST,
            dummy=DUMMY,
            return_skipped=True,
        )

        skipped = testing_process.test(models_project_name)
        if CLEAR_OUTPUT:
            clear_output()
        
        if skipped != {}:
            print('Skipped:')
            for key, val in skipped.items():
                print(f'{key}: {val}')
    else:
        print('WARNING! testing is disabled')

    finished()
    
except Exception as e:
    print('Interrupt:\n')
    traceback.print_exc()

    
    
# ================================================================================
if not is_user_done(USER_ID):
    %notify -m "Training Process: INTERRUPTED"