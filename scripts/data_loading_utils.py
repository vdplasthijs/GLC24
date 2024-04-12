import os, sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
from loadpaths_glc import loadpaths
path_dict = loadpaths()

def load_metadata():
    df_train_pa = pd.read_csv(os.path.join(path_dict['data_folder'], 'GLC24_PA_metadata_train.csv'))
    df_train_po = pd.read_csv(os.path.join(path_dict['data_folder'], 'GLC24_P0_metadata_train.csv'))
    df_test_pa = pd.read_csv(os.path.join(path_dict['data_folder'], 'GLC24_PA_metadata_test.csv'))

    cols_drop = ['taxonRank', 'geoUncertaintyInM', 'date', 'areaInM2']
    print(f'Columns dropped: {cols_drop}')
    for df in [df_train_pa, df_train_po, df_test_pa]:
        for c in cols_drop:
            if c in df.columns:
                df.drop(c, axis=1, inplace=True)

    dict_dfs = {'df_train_pa': df_train_pa, 'df_train_po': df_train_po, 'df_test_pa': df_test_pa}
    dict_dfs_train = {'df_train_pa': df_train_pa, 'df_train_po': df_train_po}
    return dict_dfs, dict_dfs_train

def load_landsat_timeseries(mode='train', data_type='PA'):
    assert mode in ['train', 'test'], mode 
    assert data_type in ['PA'], data_type

    path_folder = os.path.join(path_dict['data_folder'], f'{data_type}-{mode}-landsat_time_series')
    names_bands = ['blue', 'green', 'nir', 'red', 'swir1', 'swir2']
    path_bands = {band: os.path.join(path_folder, f'GLC24-{data_type}-{mode}-landsat_time_series-{band}.csv') for band in names_bands}

    dict_dfs = {band: pd.read_csv(path_bands[band]) for band in names_bands}
    return dict_dfs

def load_multiple_env_raster(mode='train', data_type='PA',
                             list_env_types=['elevation', 'landcover', 'soilgrids', 'climate_av']):
    assert mode in ['train', 'test'], mode
    assert data_type in ['PA'], data_type
    
    for i_env, env_type in enumerate(list_env_types):
        df_raster = load_env_raster(env_type=env_type, mode=mode, data_type=data_type)
        if i_env == 0:
            df_merged = df_raster
        else:
            df_merged = df_merged.merge(df_raster, on='surveyId', how='inner')
    return df_merged

def load_env_raster(env_type='elevation', mode='train', data_type='PA'):
    assert mode in ['train', 'test'], mode 
    assert data_type in ['PA'], data_type
    assert env_type in ['elevation', 'human_footprint', 'landcover', 'soilgrids', 'climate_av', 'climate_monthly']

    base_path = os.path.join(path_dict['data_folder'], 'EnvironmentalRasters/EnvironmentalRasters/')
    if env_type == 'human_footprint':
        path_raster = os.path.join(base_path, f'Human Footprint/GLC24-{data_type}-{mode}-human_footprint.csv')
    elif env_type == 'climate_av':
        path_raster = os.path.join(base_path, f'Climate/Average 1981-2010/GLC24-{data_type}-{mode}-bioclimatic.csv')
    elif env_type == 'climate_monthly':
        path_raster = os.path.join(base_path, f'Climate/Monthly/GLC24-{data_type}-{mode}-bioclimatic_monthly.csv')
    else:
        folder_name = env_type[0].upper() + env_type[1:]
        path_raster = os.path.join(base_path, f'{folder_name}/GLC24-{data_type}-{mode}-{env_type}.csv')

    assert os.path.exists(path_raster), f'Path does not exist: {path_raster}'
    df_raster = pd.read_csv(path_raster)
    return df_raster

def get_path_sat_patch_per_survey(surveyId=1986, mode='train', data_type='PA'):
    assert mode in ['train', 'test'], mode 
    assert type(surveyId) == int, surveyId
    assert data_type == 'PA', 'Only PA data is available'

    mode_name = mode[0].upper() + mode[1:]
    folder_rgb = os.path.join(path_dict['data_folder'], f'{data_type}_{mode_name}_SatellitePatches_RGB/{data_type.lower()}_{mode}_patches_rgb')
    folder_nir = os.path.join(path_dict['data_folder'], f'{data_type}_{mode_name}_SatellitePatches_NIR/{data_type.lower()}_{mode}_patches_nir')

    surveyId_str = str(surveyId)
    if surveyId < 1000:
        surveyId_str = surveyId_str.zfill(4)

    ## digits follow XXXXABCD
    digits_AB = surveyId_str[-4:-2]
    digits_CD = surveyId_str[-2:]
    path_rgb = os.path.join(folder_rgb, digits_CD, digits_AB, surveyId_str + '.jpeg')
    path_nir = os.path.join(folder_nir, digits_CD, digits_AB, surveyId_str + '.jpeg')

    return path_rgb, path_nir