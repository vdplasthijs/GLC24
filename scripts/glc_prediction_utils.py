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
                     method='all_nearby_species', eval_mode='test'):
    df_env_train = dlu.load_multiple_env_raster(mode='train', list_surveyIds=dict_dfs['df_train_pa']['surveyId'].unique())
    if eval_mode == 'test':
        name_df = 'df_test_pa'
        df_env_test = dlu.load_multiple_env_raster(mode='test')
    elif eval_mode == 'val':
        name_df = 'df_val_pa'
        df_env_test = dlu.load_multiple_env_raster(mode='train', list_surveyIds=dict_dfs['df_val_pa']['surveyId'].unique())
    else:
        raise ValueError(f'Unknown eval_mode: {eval_mode}')

    df_train = pd.merge(dict_dfs['df_train_pa'], df_env_train, on='surveyId')
    df_train_species = dict_dfs_species['df_train_pa_species']
    assert np.all(df_env_test['surveyId'].values == dict_dfs[name_df]['surveyId'].values), 'SurveyId mismatch between test and env'

    if method == 'lc-specific':
        tmp2 = df_train_species.value_counts('surveyId')
        tmp = dlu.load_multiple_env_raster(mode='train')
        tmp = tmp[['surveyId', 'LandCover']]
        lc_av_sp_count = pd.merge(tmp, tmp2, on='surveyId').drop('surveyId', axis=1).groupby('LandCover').mean().reset_index()
        lc_av_sp_count['count'] = lc_av_sp_count['count'].apply(lambda x: int(np.ceil(x)))
        print(lc_av_sp_count)

    ## predictions:
    dict_pred = {}
    count_no_nearby = 0
    max_it_while_loop = 15
    for it in tqdm(range(len(dict_dfs[name_df]))):
        row = dict_dfs[name_df].iloc[it]
        point_loc = row.geometry
        curr_test_survey_id = row.surveyId
        curr_test_lc = df_env_test['LandCover'].iloc[it]    
        curr_n_sp = lc_av_sp_count[lc_av_sp_count['LandCover'] == curr_test_lc]['count'].values[0]
            
        curr_it_wh = 0 
        df_train_nearby = pd.DataFrame()
        # while curr_it_wh < max_it_while_loop and len(df_train_nearby) == 0:
        while curr_it_wh < max_it_while_loop and len(df_train_nearby) < curr_n_sp:
            curr_buffer_deg = buffer_deg * (1 + curr_it_wh)
            circle = point_loc.buffer(curr_buffer_deg) ## buffer to degrees
            nearby_training_points = df_train.sindex.intersection(circle.bounds)
            df_train_nearby = df_train.iloc[nearby_training_points]
            if len(df_train_nearby) > 0:
                df_train_nearby = df_train_nearby[df_train_nearby['LandCover'] == curr_test_lc]
            curr_it_wh += 1
        if len(df_train_nearby) == 0:
            print(f'No nearby points for surveyId: {curr_test_survey_id} (LC: {curr_test_lc})')
            dict_pred[curr_test_survey_id] = []
            count_no_nearby += 1
            continue
        nearby_survey_ids = df_train_nearby['surveyId'].unique()
        assert len(nearby_survey_ids) > 0, 'No nearby survey ids found'
        df_nearby_species = df_train_species[np.isin(df_train_species['surveyId'], nearby_survey_ids)]
        
        if method == 'all_nearby_species':
            curr_species_pred = list(df_nearby_species['speciesId'].unique())
        elif method == 'lc-specific':
            curr_species_pred = df_nearby_species['speciesId'].value_counts()[:curr_n_sp].index.tolist()
        elif 'top_' in method:
            n_top = int(method.split('_')[-1])
            curr_species_pred = df_nearby_species['speciesId'].value_counts()[:n_top].index.tolist()
            # print(len(df_nearby_species), (df_nearby_species['speciesId'].value_counts() > 1).sum())
        else:
            raise ValueError(f'Unknown method: {method}')
        dict_pred[curr_test_survey_id] = list(np.sort(curr_species_pred))

    print(f'Predictions done ({it} total). No nearby points: {count_no_nearby}.')

    if save_pred and eval_mode == 'test':
        convert_dict_pred_to_csv(dict_pred, save=True, custom_name=f'buffer-lc-{buffer_deg}-{method}')

    return dict_pred

def compute_f1_score_dicts(dict_val, dict_pred):
    assert len(dict_val) == len(dict_pred), f'Length mismatch: {len(dict_val)} vs {len(dict_pred)}'
    list_scores = []
    for k, v_val in dict_val.items():
        assert k in dict_pred, f'Key not found in dict_pred: {k}'
        v_pred = dict_pred[k]
        n_tp = len(np.intersect1d(v_val, v_pred))
        n_fp = len(np.setdiff1d(v_pred, v_val))
        n_fn = len(np.setdiff1d(v_val, v_pred))
        score = n_tp / ( n_tp + 0.5 * (n_fp + n_fn))
        list_scores.append(score)
    return np.mean(list_scores)