import os, sys, json, datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime 
import geopandas as gpd
import pandas as pd 
import scipy.sparse as sp
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
        curr_n_sp = 25
        curr_it_wh = 0 
        df_train_nearby = pd.DataFrame()
        # while curr_it_wh < max_it_while_loop and len(df_train_nearby) == 0:
        while curr_it_wh < max_it_while_loop and len(df_train_nearby) < curr_n_sp:
            curr_buffer_deg = buffer_deg * (1 + curr_it_wh)
            circle = point_loc.buffer(curr_buffer_deg) ## buffer to degrees
            nearby_training_points = df_train.sindex.intersection(circle.bounds)
            # return nearby_training_points, df_train, point_loc
            df_train_nearby = df_train.iloc[nearby_training_points]
            if len(df_train_nearby) > 0:
                df_train_nearby = df_train_nearby[df_train_nearby['LandCover'] == curr_test_lc]
            curr_it_wh += 1
            # return df_train_nearby
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

class LabelPropagation():
    def __init__(self, val_or_test='val', n_iter=10, 
                 list_env_types=['elevation', 'landcover', 'climate_av'],
                 path_inds_val=None, dist_neigh_meter=10000,
                 method_weights='dist_exp_decay',
                 preload_data=True, preload_timestamp=''):
        self.val_or_test = val_or_test
        self.n_iter = n_iter
        self.list_env_types = list_env_types
        self.path_inds_val = path_inds_val
        self.dist_neigh_meter = dist_neigh_meter
        self.method_weights = method_weights
        self.preload_data = preload_data
        self.data_folder_sparse = os.path.join(path_dict['data_folder'], 'sparse_format')
        self.preload_timestamp = preload_timestamp
        self.filter_lc_exact_match = True
        self.load_data()
        self.create_graph()

    def load_data(self):
        (df_train, df_test), (df_train_species, df_val_species) = dlu.create_full_pa_ds(
            list_env_types=self.list_env_types, val_or_test=self.val_or_test, 
            path_inds_val=self.path_inds_val, drop_surveyId=False,
            create_geo=True
        )
        self.df_train = df_train
        self.df_train_species = df_train_species
        self.df_test = df_test
        self.df_val_species = df_val_species
        if self.val_or_test == 'val':
            assert len(df_val_species) > 0, 'No validation species data'
        elif self.val_or_test == 'test':
            assert df_val_species is None, 'Validation species data found'

    def create_graph(self):
        assert 'surveyId' in self.df_train.columns, 'surveyId not in df_train'
        if self.val_or_test == 'val':
            assert 'surveyId' in self.df_test.columns, 'surveyId not in df_test'

        ## change to crs 3857 for distance calculations
        self.df_train = self.df_train.to_crs(3857)
        self.df_test = self.df_test.to_crs(3857)
        assert self.df_train.columns.equals(self.df_test.columns)
        self.df_features_merged = pd.concat([self.df_train, self.df_test])
        self.df_features_merged = self.df_features_merged.reset_index(drop=True)
        self.n_train = len(self.df_train)
        assert self.df_features_merged.iloc[:self.n_train].equals(self.df_train), 'Mismatch in df_train'
        # assert self.df_features_merged.iloc[self.n_train:].equals(self.df_test), 'Mismatch in df_test'
        # assert self.df_features_merged.iloc[self.n_train:].equals(self.df_test), 'Mismatch in df_test'
        self.n_samples = len(self.df_features_merged)
        if self.val_or_test == 'val':
            self.df_species_merged = pd.concat([self.df_train_species, self.df_val_species])
        else:
            self.df_species_merged = self.df_train_species
        self.n_species = self.df_species_merged['speciesId'].nunique()
        self.species_array = self.df_species_merged['speciesId'].unique()
        self.dict_species_val_to_ind = {sp: i for i, sp in enumerate(self.species_array)}
        self.dict_species_ind_to_val = {i: sp for i, sp in enumerate(self.species_array)}
        self.dict_surveys_val_to_ind = {surveyId: i for i, surveyId in enumerate(self.df_features_merged['surveyId'])}

        if self.preload_data is False:
            assert self.preload_timestamp is None or self.preload_timestamp == '', 'preload_timestamp not None or empty'
            self.preload_timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')

        path_sparse_labels = os.path.join(self.data_folder_sparse, f'mat_labels_{self.val_or_test}_{self.preload_timestamp}.npz')
        path_sparse_weights = os.path.join(self.data_folder_sparse, f'mat_weights_{self.val_or_test}_{self.preload_timestamp}.npz')
        path_sparse_dist = os.path.join(self.data_folder_sparse, f'mat_dist_{self.val_or_test}_{self.preload_timestamp}.npz')
        path_sparse_edges = os.path.join(self.data_folder_sparse, f'mat_edges_{self.val_or_test}_{self.preload_timestamp}.npz')
        path_metadata = os.path.join(self.data_folder_sparse, f'metadata_{self.val_or_test}_{self.preload_timestamp}.json')

        if self.preload_data:
            assert os.path.exists(path_sparse_labels) 
            assert os.path.exists(path_sparse_weights) 
            assert os.path.exists(path_sparse_dist) 
            assert os.path.exists(path_sparse_edges)
            
            self.mat_labels = sp.load_npz(path_sparse_labels)
            self.mat_weights = sp.load_npz(path_sparse_weights)
            self.mat_dist = sp.load_npz(path_sparse_dist)
            self.mat_edges = sp.load_npz(path_sparse_edges)
            with open(path_metadata, 'r') as f:
                self.dict_metadata = json.load(f)
            print('Loaded sparse matrices from file')
            return None

        ## Create sparse label matrix:
        print('Creating sparse label matrix')
        self.mat_labels = sp.lil_matrix((self.n_samples, self.n_species), dtype=np.float32)
        for row, surveyId in tqdm(enumerate(self.df_train['surveyId'])):
            assert row == self.dict_surveys_val_to_ind[surveyId], 'Row mismatch'
            curr_species = self.df_train_species.loc[self.df_train_species['surveyId'] == surveyId]
            curr_species = curr_species['speciesId'].map(self.dict_species_val_to_ind).values
            self.mat_labels[row, curr_species] = 1

        assert row == self.n_train - 1, 'Row mismatch'

        ## Create sparse weight matrix:
        print('Creating sparse weight matrix')
        self.mat_weights = sp.lil_matrix((self.n_samples, self.n_samples), dtype=np.float32)
        self.mat_dist = sp.lil_matrix((self.n_samples, self.n_samples), dtype=np.float32)
        self.mat_edges = sp.lil_matrix((self.n_samples, self.n_samples), dtype=bool)
        for row, surveyId in tqdm(enumerate(self.df_features_merged['surveyId'])):
            assert row == self.dict_surveys_val_to_ind[surveyId], 'Row mismatch'
            point_loc = self.df_features_merged.loc[self.df_features_merged['surveyId'] == surveyId].geometry.values[0]
            current_lc = self.df_features_merged.loc[self.df_features_merged['surveyId'] == surveyId].LandCover.values[0]
            circle = point_loc.buffer(self.dist_neigh_meter)
            nearby_points = self.df_features_merged.sindex.intersection(circle.bounds)
            nearby_points = np.setdiff1d(nearby_points, row)  ## remove self
            if self.filter_lc_exact_match:
                nearby_points_lc_match = self.df_features_merged.iloc[nearby_points]
                nearby_points_lc_match = nearby_points_lc_match[nearby_points_lc_match['LandCover'] == current_lc]
                nearby_points = nearby_points_lc_match.index

            dist_to_points = self.df_features_merged.iloc[nearby_points].distance(point_loc)
            self.mat_dist[row, nearby_points] = dist_to_points
            self.mat_edges[row, nearby_points] = 1
            if self.method_weights == 'dist_exp_decay':
                self.mat_weights[row, nearby_points] = np.exp(-dist_to_points / self.dist_neigh_meter)
            else:
                raise ValueError(f'Unknown method_weights: {self.method_weights}')

        sp.save_npz(path_sparse_labels, self.mat_labels.tocsr())
        sp.save_npz(path_sparse_weights, self.mat_weights.tocsr())
        sp.save_npz(path_sparse_dist, self.mat_dist.tocsr())
        sp.save_npz(path_sparse_edges, self.mat_edges.tocsr())

        self.dict_metadata = {
            'n_samples': self.n_samples,
            'n_species': self.n_species,
            'n_train': self.n_train,
            'list_env_types': self.list_env_types,
            'dist_neigh_meter': self.dist_neigh_meter,
            'filter_lc_exact_match': self.filter_lc_exact_match,
            'method_weights': self.method_weights,
            'preload_timestamp': self.preload_timestamp
        }
        with open(path_metadata, 'w') as f:
            json.dump(self.dict_metadata, f)

        print('Saved sparse matrices to file')
        return None 


    def fit(self):
        ## Create sparse label matrix:
        array_diffs = []
        diff_threshold = 1

        self.mat_labels_fit = self.mat_labels.copy()
        sum_weights = self.mat_weights.sum(axis=1)[self.n_train:]
        for it in range(self.n_iter):
            mat_labels_new_test = self.mat_weights[self.n_train:, :] @ self.mat_labels_fit
            mat_labels_new_test = mat_labels_new_test / sum_weights
            diff = mat_labels_new_test - self.mat_labels_fit[self.n_train:, :]
            diff_nz = diff[diff.nonzero()]
            array_diffs.append(abs(diff_nz).sum())
            self.mat_labels_fit[self.n_train:, :] = mat_labels_new_test
            print(f'Iteration {it + 1}/{self.n_iter}. Diff: {array_diffs[-1]}')
            if array_diffs[-1] < diff_threshold:
                break

        print(f'Converged after {it + 1}/{self.n_iter} iterations')

        return array_diffs

    def create_predictions(self, threshold_method='fixed', threshold_weighted_labels=0.1, save_pred=True):
        assert threshold_method in ['adaptive', 'fixed'], f'Unknown threshold_method: {threshold_method}'
        self.dict_pred = {}
        assert hasattr(self, 'mat_labels_fit'), 'mat_labels_fit not found'
        count_no_species = 0
        self.mat_labels_thresholded = self.mat_labels_fit.copy()

        if threshold_method == 'adaptive':
            n_test = len(self.df_test)
            assert self.n_train + n_test == self.mat_labels_fit.shape[0], 'Mismatch in mat_labels_fit'
            size_pos_labels_train = np.squeeze(np.array(sp.csc_matrix(self.mat_labels_fit[:self.n_train, :]).sum(axis=0)))
            size_pos_labels_target = size_pos_labels_train * (n_test / self.n_train)
            assert self.n_species == len(size_pos_labels_train), 'Mismatch in size_pos_labels_train'
            thresholds_test = np.zeros(self.n_species)
            mat_pred_csc = sp.csc_matrix(self.mat_labels_fit[self.n_train:, :])  # for efficient column slicing
            for species_ind in range(self.n_species):
                curr_labels = mat_pred_csc[:, species_ind]
                sorted_preds = np.sort(curr_labels.toarray().flatten())
                curr_target_size = size_pos_labels_target[species_ind]
                assert curr_target_size > 0 and curr_target_size < self.n_species, f'size_pos_labels_target is 0: {curr_target_size}'
                thresholds_test[species_ind] = sorted_preds[-int(np.ceil(curr_target_size))]  ## ceil first to prevent rounding to 0 for target sizes < 1. 
                if thresholds_test[species_ind] == 0:  ## this shouldn't matter, because th==0 only if all preds are 0, but just in case to prevent future blow ups.
                    thresholds_test[species_ind] = 0.01
                # else:
                #     thresholds_test[species_ind] = np.minimum(thresholds_test[species_ind], threshold_weighted_labels)

            print(f'Computed thresholds for test set')
        elif threshold_method == 'fixed':
            thresholds_test = threshold_weighted_labels
        else:
            raise ValueError(f'Unknown threshold_method: {threshold_method}')
        self.thresholds_test = thresholds_test

        for surveyId in tqdm(self.df_test['surveyId']):
            row = self.dict_surveys_val_to_ind[surveyId]
            curr_labels = self.mat_labels_fit[row, :]
            if curr_labels.sum() == 0:
                self.dict_pred[surveyId] = []
                count_no_species += 1
                continue

            weighted_labels_thresholded = curr_labels > thresholds_test
            self.mat_labels_thresholded[row, :] = weighted_labels_thresholded
            if weighted_labels_thresholded.sum() == 0:
                self.dict_pred[surveyId] = []
                count_no_species += 1
                continue
            self.dict_pred[surveyId] = [self.dict_species_ind_to_val[i] for i in weighted_labels_thresholded.nonzero()[1]]
        
        print(f'Predictions done. No species found: {count_no_species}/{len(self.df_test)}.')
        if save_pred and self.val_or_test == 'test':
            convert_dict_pred_to_csv(self.dict_pred, save=True, 
                                     custom_name=f'label-prop-lc-{self.dist_neigh_meter}m-{threshold_weighted_labels}')
        return self.dict_pred
        
    def compute_f1_score_pred(self):
        if self.df_val_species is None:
            print('No validation data available')
            return None
        
        dict_val = {}
        for surveyId in self.df_val_species['surveyId'].unique():
            curr_species = self.df_val_species.loc[self.df_val_species['surveyId'] == surveyId]
            curr_species = curr_species['speciesId'].values
            dict_val[surveyId] = curr_species

        return compute_f1_score_dicts(dict_val, self.dict_pred)