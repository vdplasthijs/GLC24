import os, sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime 
from loadpaths_glc import loadpaths
path_dict = loadpaths()

def convert_dict_pred_to_csv(dict_pred, save=True, custom_name=''):
    assert len(dict_pred) == 4716, f'Not expected len for GLC 2024: {len(dict_pred)}'
    for k, v in dict_pred.items():
        assert type(k) == int, f'Key is not int: {k}'
        assert type(v) == list, f'Value is not list: {v}'
        dict_pred[k] = list(np.sort(v))
        ## assert elements in v are ints
        # for i in v:
        #     assert type(i) == int, f'Element in value is not int: {i}'

    list_keys = list(dict_pred.keys())
    list_vals = list(dict_pred.values())
    df = pd.DataFrame({'surveyId': list_keys, 'predictions': list_vals})
    df['predictions'] = df['predictions'].apply(lambda x: ' '.join(map(str, x)).lstrip('[ ').rstrip(' ]').replace("\n", ""))

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    path_save = os.path.join(path_dict['predictions_folder'], f'GLC24_vdplasthijs_predictions-{custom_name}_{timestamp}.csv')
    if save:
        df.to_csv(path_save, index=False)
        print(f'Saved predictions to: {path_save}')
    return df