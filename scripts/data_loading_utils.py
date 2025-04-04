import os, sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
from loadpaths_glc import loadpaths
import geopandas as gpd
from shapely.geometry import Point
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

path_dict = loadpaths()
GLC_YEAR = '25'

def load_metadata(create_geo=False, add_h3=False, drop_po=False,
                  drop_duplicates=True, create_validation_set=False, 
                  path_inds_val=None):
    if add_h3 is not False:
        assert type(add_h3) == int, add_h3
        bool_add_h3 = True
    else:
        bool_add_h3 = False
    if create_geo and drop_po is False:
        drop_po = True
        print('Dropped PO data because takes ages with geometry')
    df_train_pa = pd.read_csv(os.path.join(path_dict['data_folder'], f'GLC{GLC_YEAR}_PA_metadata_train.csv'))
    df_train_po = pd.read_csv(os.path.join(path_dict['data_folder'], f'GLC{GLC_YEAR}_P0_metadata_train.csv'))
    df_test_pa = pd.read_csv(os.path.join(path_dict['data_folder'], f'GLC{GLC_YEAR}_PA_metadata_test.csv'))

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
    
    if drop_duplicates:
        for k, v in dict_dfs_species.items():
            n_entries = len(v)
            dict_dfs_species[k] = v.drop_duplicates()
            n_entries_after = len(dict_dfs_species[k])
            print(f'Dropped {n_entries - n_entries_after}/{n_entries} duplicates in {k}')

    if bool_add_h3:
        for k, v in dict_dfs.items():
            dict_dfs[k] = v.h3.geo_to_h3(resolution=add_h3)
            dict_dfs[k] = dict_dfs[k].reset_index()
            
    if create_geo:
        for k, v in dict_dfs.items():
            v['geometry'] = [Point(xy) for xy in zip(v.lng, v.lat)]
            dict_dfs[k] = gpd.GeoDataFrame(v, crs='EPSG:4326')
    
    if create_validation_set:
        if path_inds_val is None:
            if GLC_YEAR == '24':
                path_inds_val = '../content/val_inds/inds_val_pa-20240416-1454.npy'
            else:
                print('Creating new validation set')
                fraction_val_pa = 0.1
                n_val_pa = int(fraction_val_pa * len(dict_dfs['df_train_pa']))
                inds_val_pa = np.random.choice(dict_dfs['df_train_pa'].index, n_val_pa, replace=False)
                timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M')
                path_inds_val = os.path.join('../content/val_inds/', f'inds_val_pa-{timestamp}.npy')
                np.save(path_inds_val, inds_val_pa)
        assert os.path.exists(path_inds_val), f'Path does not exist: {path_inds_val}'
        assert path_inds_val.endswith('.npy'), path_inds_val
        inds_val_pa = np.load(path_inds_val)
        dict_dfs['df_val_pa'] = dict_dfs['df_train_pa'].loc[inds_val_pa].sort_values('surveyId')   
        dict_dfs['df_train_pa'] = dict_dfs['df_train_pa'].drop(inds_val_pa)

        dict_dfs_species['df_val_pa_species'] = dict_dfs_species['df_train_pa_species'][np.isin(dict_dfs_species['df_train_pa_species']['surveyId'], dict_dfs['df_val_pa']['surveyId'])].sort_values('surveyId')
        dict_dfs_species['df_train_pa_species'] = dict_dfs_species['df_train_pa_species'][~np.isin(dict_dfs_species['df_train_pa_species']['surveyId'], dict_dfs['df_val_pa']['surveyId'])]
        print(f'Created validation set with {len(dict_dfs["df_val_pa"])} entries')   

        dict_val_species = {}
        for row in dict_dfs_species['df_val_pa_species'].itertuples():
            surveyId = row.surveyId
            speciesId = row.speciesId
            if surveyId not in dict_val_species:
                dict_val_species[surveyId] = [speciesId]
            else:
                dict_val_species[surveyId].append(speciesId)

        for k, v in dict_val_species.items():
            dict_val_species[k] = list(np.unique(v))

    else:
        dict_val_species = None

    return dict_dfs, dict_dfs_species, dict_val_species

def clean_po_data(dict_dfs, dict_df_species):
    assert 'df_train_po' in dict_dfs, 'No PO data available'
    assert 'df_train_pa' in dict_dfs, 'No PA data available'
    assert 'df_test_pa' in dict_dfs, 'No test data available'
    assert 'df_train_pa_species' in dict_df_species, 'No PA species data available'
    assert 'df_train_po_species' in dict_df_species, 'No PO species data available'
    ## filter locations too far away from test data
    pass 

    ## filter species not present in train pa data (?)
    pass

    return dict_dfs, dict_df_species

def load_landsat_timeseries(mode='train', data_type='PA'):
    assert mode in ['train', 'test'], mode 
    assert data_type in ['PA'], data_type

    if GLC_YEAR == '24':
        path_folder = os.path.join(path_dict['data_folder'], f'{data_type}-{mode}-landsat_time_series')
    elif GLC_YEAR == '25':
        path_folder = os.path.join(path_dict['data_folder'], 'SateliteTimeSeries-Landsat/values/')
    names_bands = ['blue', 'green', 'nir', 'red', 'swir1', 'swir2']
    path_bands = {band: os.path.join(path_folder, f'{data_type}-{mode}/GLC{GLC_YEAR}-{data_type}-{mode}-landsat_time_series-{band}.csv') for band in names_bands}

    dict_dfs = {band: pd.read_csv(path_bands[band]) for band in names_bands}

    ## merge all dfs, drop nan cols:
    for it, (key, val_df) in enumerate(dict_dfs.items()):
        col_inds_nan = np.where(val_df.isna().sum(0) > 0)[0]
        assert col_inds_nan[0] == 73
        val_df = val_df.drop(columns=val_df.columns[col_inds_nan])
        ## rename cols:
        val_df = val_df.rename(columns={c: f'{c}_{key}' for c in val_df.columns if c != 'surveyId'})
        if it == 0:
            df_all = val_df
        else:
            df_all = df_all.merge(val_df, on='surveyId', how='outer')
    assert df_all.isna().sum().sum() == 0
    return dict_dfs, df_all

def load_multiple_env_raster(mode='train', data_type='PA', list_surveyIds=None,
                             list_env_types=['elevation', 'landcover', 'climate_av']):
    assert mode in ['train', 'test'], mode
    assert data_type in ['PA'], data_type
    if 'soilgrids' in list_env_types:
        print('WARNING: soil grids contain many nans')
    
    for i_env, env_type in enumerate(list_env_types):
        if env_type == 'landsat':
            _, df_raster = load_landsat_timeseries(mode=mode, data_type=data_type)
        else:
            df_raster = load_env_raster(env_type=env_type, mode=mode, data_type=data_type)
        if i_env == 0:
            df_merged = df_raster
        else:
            df_merged = df_merged.merge(df_raster, on='surveyId', how='inner')

    if list_surveyIds is not None:
        df_merged = df_merged[df_merged['surveyId'].isin(list_surveyIds)]

    return df_merged

def load_env_raster(env_type='elevation', mode='train', data_type='PA'):
    assert mode in ['train', 'test'], mode 
    assert data_type in ['PA'], data_type
    assert env_type in ['elevation', 'human_footprint', 'landcover', 'soilgrids', 'climate_av', 'climate_monthly']

    if GLC_YEAR == '24':
        base_path = os.path.join(path_dict['data_folder'], 'EnvironmentalRasters/EnvironmentalRasters/')
    elif GLC_YEAR == '25':
        base_path = os.path.join(path_dict['data_folder'], 'EnvironmentalValues/')
    if env_type == 'human_footprint':
        path_raster = os.path.join(base_path, f'Human Footprint/GLC{GLC_YEAR}-{data_type}-{mode}-human_footprint.csv')
    elif env_type == 'climate_av':
        if GLC_YEAR == '24':
            path_raster = os.path.join(base_path, f'Climate/Average 1981-2010/GLC{GLC_YEAR}-{data_type}-{mode}-bioclimatic.csv')
        elif GLC_YEAR == '25':
            path_raster = os.path.join(base_path, f'ClimateAverage_1981-2010/GLC{GLC_YEAR}-{data_type}-{mode}-bioclimatic.csv')
    elif env_type == 'climate_monthly':
        path_raster = os.path.join(base_path, f'Climate/Monthly/GLC{GLC_YEAR}-{data_type}-{mode}-bioclimatic_monthly.csv')
    elif env_type == 'landcover':
        if GLC_YEAR == '24':
            path_raster = os.path.join(base_path, f'Landcover/GLC{GLC_YEAR}-{data_type}-{mode}-landcover.csv')
        elif GLC_YEAR == '25':
            path_raster = os.path.join(base_path, f'LandCover/GLC{GLC_YEAR}-{data_type}-{mode}-landcover.csv')
    else:
        folder_name = env_type[0].upper() + env_type[1:]
        path_raster = os.path.join(base_path, f'{folder_name}/GLC{GLC_YEAR}-{data_type}-{mode}-{env_type}.csv')

    assert os.path.exists(path_raster), f'Path does not exist: {path_raster}'
    df_raster = pd.read_csv(path_raster)

    if env_type == 'elevation':
        n_nans = df_raster.isna().sum().sum()
        if data_type == 'PO':
            assert n_nans == 0, 'Nans in elevation data'
        else:
            if mode == 'test':
                assert n_nans == 0, 'Nans in elevation data'
            elif mode == 'train':
                assert n_nans == 35
                print(f'Nans in elevation data: {n_nans}. Manually fixing now.')

        ## Look up elevation using lon/lat (offline): https://www.freemaptools.com/elevation-finder.htm
        dict_elevation_fill = {
            168459: 13,
            178182: 12,
            249903: 13,
            509739: 408,
            634794: 13,
            885566: 13,
            992568: 0,
            1123670: 56,
            1128141: 55,
            1154317: 13,
            1176780: 103,
            1265195: 103,
            1396637: 103,
            1504221: 323,
            1658226: 13,
            1718915: 13,
            1738571: 13,
            1934450: 13,
            1967118: 13,
            2006152: 41,
            2007673: 12,
            2211670: 13,
            2329898: 13,
            2367983: 13,
            2499157: 13,
            2616656: 13,
            2619214: 13,
            2651568: 13,
            2760392: 13,
            2775439: 90,
            2863281: 13,
            3026478: 13,
            3193990: 13,
            3563432: 13,
            3705072: 13
        }
        for surveyId, elevation in dict_elevation_fill.items():
            df_raster.loc[df_raster['surveyId'] == surveyId, 'Elevation'] = elevation
        assert df_raster.isna().sum().sum() == 0, 'Nans in elevation data'
        print('Nans in elevation data fixed')
    return df_raster

def get_path_sat_patch_per_survey(surveyId=1986, mode='train', data_type='PA'):
    assert mode in ['train', 'test'], mode 
    assert type(surveyId) == int, surveyId
    assert data_type == 'PA', 'Only PA data is available'

    mode_name = mode[0].upper() + mode[1:]
    folder_rgb = os.path.join(path_dict['data_folder'], f'{data_type}_{mode_name}_SatellitePatches_RGB/{data_type.lower()}_{mode}_patches_rgb')
    folder_nir = os.path.join(path_dict['data_folder'], f'{data_type}_{mode_name}_SatellitePatches_NIR/{data_type.lower()}_{mode}_patches_nir')

    ## digits follow XXXXABCD
    surveyId_str = str(surveyId)
    if surveyId < 1000:
        assert surveyId >= 100, f'{surveyId} is below 100, double check how folder structure is created'
        # surveyId_str = surveyId_str.zfill(4)
        digits_AB = surveyId_str[0]
    else:
        digits_AB = surveyId_str[-4:-2]
    digits_CD = surveyId_str[-2:]
    path_rgb = os.path.join(folder_rgb, digits_CD, digits_AB, surveyId_str + '.jpeg')
    path_nir = os.path.join(folder_nir, digits_CD, digits_AB, surveyId_str + '.jpeg')

    return path_rgb, path_nir

def load_sat_patch(surveyId=212, mode='train', data_type='PA'):
    path_rgb, path_nir = get_path_sat_patch_per_survey(surveyId=surveyId, mode=mode, data_type=data_type)
    assert os.path.exists(path_rgb), f'Path does not exist: {path_rgb}'
    img_rgb = np.array(Image.open(path_rgb))
    img_nir = np.array(Image.open(path_nir))
    assert img_rgb.shape == (128, 128, 3), f'Image shape is {img_rgb.shape}'
    assert img_nir.shape == (128, 128), f'Image shape is {img_nir.shape}'
    img_comb = np.concatenate([img_rgb, img_nir[..., np.newaxis]], axis=-1)
    return img_comb

def create_full_pa_ds(list_env_types=['elevation', 'landcover', 'climate_av'],
                      drop_surveyId=True, val_or_test='val', path_inds_val=None,
                      create_geo=False, transform_pca=False, pca_threshold=0.9):
    '''If val_or_test is "val", then the validation set is created (incl species), otherwise the test set is created (with no species).'''
    assert val_or_test in ['val', 'test'], val_or_test
    dict_dfs, dict_dfs_species, _ = load_metadata(create_geo=create_geo, add_h3=False, path_inds_val=path_inds_val,
                                                  create_validation_set=True if val_or_test == 'val' else False)
    
    ## Create train set:
    df_env_train = load_multiple_env_raster(mode='train', list_surveyIds=dict_dfs['df_train_pa']['surveyId'].unique(),
                                            list_env_types=list_env_types) 
    if transform_pca:
        scaler = StandardScaler()
        assert df_env_train.columns[0] == 'surveyId', 'First column is not surveyId'
        array_surveyid = df_env_train['surveyId'].values
        tmp_vals = df_env_train.values[:, 1:]
        tmp_vals = scaler.fit_transform(tmp_vals)
        pca = PCA(n_components=pca_threshold)
        tmp_vals = pca.fit_transform(tmp_vals)
        tmp_dict = {**{'surveyId': array_surveyid}, **{f'PCA_{i}': tmp_vals[:, i] for i in range(tmp_vals.shape[1])}}
        df_env_train = pd.DataFrame(tmp_dict)
        array_expl_var = pca.explained_variance_ratio_
    else:
        array_expl_var = None
    df_train = pd.merge(dict_dfs['df_train_pa'], df_env_train, on='surveyId')
    df_train.dropna(inplace=True)
    df_train_species = dict_dfs_species['df_train_pa_species'][dict_dfs_species['df_train_pa_species']['surveyId'].isin(df_train['surveyId'])]

    ## Create val or test set:
    if val_or_test == 'val':
        df_env_test = load_multiple_env_raster(mode='train', list_surveyIds=dict_dfs['df_val_pa']['surveyId'].unique(),
                                            list_env_types=list_env_types)
        if transform_pca:
            tmp_vals = df_env_test.values[:, 1:]
            tmp_vals = scaler.transform(tmp_vals)
            tmp_vals = pca.transform(tmp_vals)
            tmp_dict = {**{'surveyId': df_env_test['surveyId'].values}, **{f'PCA_{i}': tmp_vals[:, i] for i in range(tmp_vals.shape[1])}}
            df_env_test = pd.DataFrame(tmp_dict)
        df_test = pd.merge(dict_dfs['df_val_pa'], df_env_test, on='surveyId')
        df_test.dropna(inplace=True)
        df_val_species = dict_dfs_species['df_val_pa_species'][dict_dfs_species['df_val_pa_species']['surveyId'].isin(df_test['surveyId'])]
    elif val_or_test == 'test':
        df_env_test = load_multiple_env_raster(mode='test', list_env_types=list_env_types)
        if transform_pca:
            tmp_vals = df_env_test.values[:, 1:]
            tmp_vals = scaler.transform(tmp_vals)
            tmp_vals = pca.transform(tmp_vals)
            tmp_dict = {**{'surveyId': df_env_test['surveyId'].values}, **{f'PCA_{i}': tmp_vals[:, i] for i in range(tmp_vals.shape[1])}}
            df_env_test = pd.DataFrame(tmp_dict)
        df_test = pd.merge(dict_dfs['df_test_pa'], df_env_test, on='surveyId')
        df_val_species = None

    assert df_train.isna().sum().sum() == 0, 'Nans in train data'
    assert df_test.isna().sum().sum() == 0, 'Nans in test data'
    assert df_train.columns.equals(df_test.columns), 'Columns differ between train and test'
    assert df_train.shape[1] == df_test.shape[1], 'Number of columns differ between train and test'
    if drop_surveyId:
        df_train = df_train.drop('surveyId', axis=1)
        df_test = df_test.drop('surveyId', axis=1)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    assert df_train.columns.equals(df_test.columns), 'Columns differ between train and test'

    return (df_train, df_test), (df_train_species, df_val_species), array_expl_var