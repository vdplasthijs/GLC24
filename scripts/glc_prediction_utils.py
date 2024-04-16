import os, sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime 
import geopandas as gpd
import pandas as pd 
import h3pandas 
import rtree
import data_loading_utils as dlu
from loadpaths_glc import loadpaths
path_dict = loadpaths()

def convert_dict_pred_to_csv(dict_pred, save=True, custom_name=''):
    assert len(dict_pred) == 4716, f'Not expected len for GLC 2024: {len(dict_pred)}'
    for k, v in dict_pred.items():
        assert type(k) in [int, np.int64, np.int32], f'Key is not int but {type(k)}: {k}'
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

def predict_using_buffer(dict_dfs, dict_dfs_species, buffer_deg=0.2, save_pred=False,
                     method='all_nearby_species'):
    df_env_train = dlu.load_multiple_env_raster(mode='train')
    df_env_test = dlu.load_multiple_env_raster(mode='test')
    # for c in dict_dfs['df_train_pa'].columns:
    #     if 'h3' in c:
    #         h3_col = c
    #         break
    # assert h3_col in dict_dfs['df_test_pa'].columns, f'H3 column not found in df_test_pa'
    # h3_int = int(h3_col.split('_')[-1])

    # print(f'Predicting using H3 of resolution {h3_int}')
    df_train = pd.merge(dict_dfs['df_train_pa'], df_env_train, on='surveyId')
    df_train_species = dict_dfs_species['df_train_pa_species']
    assert np.all(df_env_test['surveyId'] == dict_dfs['df_test_pa']['surveyId']), 'SurveyId mismatch between test and env'

    ## prdictions:
    dict_pred = {}
    count_no_nearby = 0
    count_no_nearby_lc = 0
    for it in tqdm(range(len(dict_dfs['df_test_pa']))):
        row = dict_dfs['df_test_pa'].iloc[it]
        point_loc = row.geometry
        curr_test_survey_id = row.surveyId
        circle = point_loc.buffer(buffer_deg) ## buffer to degrees
        nearby_training_points = df_train.sindex.intersection(circle.bounds)
        if len(nearby_training_points) == 0:
            dict_pred[curr_test_survey_id] = []
            count_no_nearby += 1
            continue 
        curr_test_lc = df_env_test['LandCover'].iloc[it]    
        df_train_nearby = df_train.iloc[nearby_training_points]
        df_train_nearby = df_train_nearby[df_train_nearby['LandCover'] == curr_test_lc]
        # print(len(nearby_training_points), len(df_train_nearby))
        if len(df_train_nearby) == 0:
            dict_pred[curr_test_survey_id] = []
            count_no_nearby_lc += 1
            continue
        nearby_survey_ids = df_train_nearby['surveyId'].unique()
        assert len(nearby_survey_ids) > 0, 'No nearby survey ids found'
        # print(nearby_survey_ids)
        df_nearby_species = df_train_species[np.isin(df_train_species['surveyId'], nearby_survey_ids)]
        
        if method == 'all_nearby_species':
            curr_species_pred = list(df_nearby_species['speciesId'].unique())
        elif method == 'top_25':
            curr_species_pred = df_nearby_species['speciesId'].value_counts()[:25].index.tolist()
            # print(len(df_nearby_species), (df_nearby_species['speciesId'].value_counts() > 1).sum())
        elif method == 'top_adaptive':
            val_counts = df_nearby_species['speciesId'].value_counts()
            val_counts = val_counts[val_counts > 1]
            n_vals = len(val_counts)
            if n_vals == 0:
                curr_species_pred = df_nearby_species['speciesId'].value_counts().index.tolist()
            else:
                curr_species_pred = val_counts.index.tolist()
        else:
            raise ValueError(f'Unknown method: {method}')
        dict_pred[curr_test_survey_id] = curr_species_pred

    print(f'Predictions done ({it} total). No nearby points: {count_no_nearby}, No nearby points with same LC: {count_no_nearby_lc}')

    if save_pred:
        convert_dict_pred_to_csv(dict_pred, save=True, custom_name=f'buffer-lc-{buffer_deg}-{method}')

    return dict_pred