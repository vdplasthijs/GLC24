import os, sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
from loadpaths_glc import loadpaths
import geopandas as gpd
from shapely.geometry import Point
import h3pandas
path_dict = loadpaths()

def load_metadata(create_geo=False, add_h3=False, drop_po=False):
    if add_h3 is not False:
        assert type(add_h3) == int, add_h3
        bool_add_h3 = True
    if create_geo and drop_po is False:
        drop_po = True
        print('Dropped PO data because takes ages with geometry')
    df_train_pa = pd.read_csv(os.path.join(path_dict['data_folder'], 'GLC24_PA_metadata_train.csv'))
    df_train_po = pd.read_csv(os.path.join(path_dict['data_folder'], 'GLC24_P0_metadata_train.csv'))
    df_test_pa = pd.read_csv(os.path.join(path_dict['data_folder'], 'GLC24_PA_metadata_test.csv'))

    cols_drop = ['taxonRank', 'geoUncertaintyInM', 'date', 'areaInM2', 'publisher',
                 'month', 'day', 'region', 'dayOfYear', 'country']  ## drop all non informative and/or non omnipresent columns
    print(f'Columns dropped: {cols_drop}')
    for df in [df_train_pa, df_train_po, df_test_pa]:
        for c in cols_drop:
            if c in df.columns:
                df.drop(c, axis=1, inplace=True)
        ## rename lon to lng
        if 'lon' in df.columns:
            df.rename(columns={'lon': 'lng'}, inplace=True)
        else:
            assert 'lng' in df.columns, 'Longitude column not found'
        if 'speciesId' in df.columns:
            df['speciesId'] = df['speciesId'].astype(int)
            
    assert set(df_train_pa.columns) == set(df_train_po.columns), f'Columns differ between train PA and PO: {df_train_pa.columns} vs {df_train_po.columns}'  
    order_cols = df_train_pa.columns
    df_train_po = df_train_po[order_cols]
    
    df_train_pa_species = df_train_pa[['speciesId', 'surveyId']]
    df_train_po_species = df_train_po[['speciesId', 'surveyId']]
    df_train_pa = df_train_pa.drop('speciesId', axis=1)
    df_train_po = df_train_po.drop('speciesId', axis=1)
    df_train_pa = df_train_pa.drop_duplicates()
    df_train_po = df_train_po.drop_duplicates()

    dict_dfs = {'df_train_pa': df_train_pa, 'df_train_po': df_train_po, 'df_test_pa': df_test_pa}
    dict_dfs_species = {'df_train_pa_species': df_train_pa_species, 'df_train_po_species': df_train_po_species}

    if drop_po:
        dict_dfs = {k: v for k, v in dict_dfs.items() if 'pa' in k}
        dict_dfs_species = {k: v for k, v in dict_dfs_species.items() if 'pa' in k}
    
    if bool_add_h3:
        for k, v in dict_dfs.items():
            dict_dfs[k] = v.h3.geo_to_h3(resolution=add_h3)
            dict_dfs[k] = dict_dfs[k].reset_index()
            
    if create_geo:
        for k, v in dict_dfs.items():
            v['geometry'] = [Point(xy) for xy in zip(v.lng, v.lat)]
            dict_dfs[k] = gpd.GeoDataFrame(v, crs='EPSG:4326')
    
    return dict_dfs, dict_dfs_species

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