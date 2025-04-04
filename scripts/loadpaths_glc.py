'''
Use by:

import loadpaths_glc
user_paths_dict = loadpath_glc.loadpaths()
'''


import os
import json
import sys
import getpass
from pathlib import Path
GLC_YEAR = '24'


def loadpaths(username=None):
    '''Function that loads data paths from json file based on computer username'''

    ## Get json:
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    json_path = os.path.join(Path(__location__).parent, 'content/data_paths_glc.json')
    json_path = str(json_path)

    if username is None:
        username = getpass.getuser()  # get username of PC account

    ## Load paths corresponding to username:
    with open(json_path, 'r') as config_file:
        config_info = json.load(config_file)
        assert username in config_info.keys(), 'Please add your username and data paths to cnn-land-cover/data_paths_glc.json'
        user_paths_dict = config_info[username]['paths']  # extract paths from current user

    # Expand tildes in the json paths
    user_paths_dict = {k: str(v) for k, v in user_paths_dict.items()}
    return {k: os.path.expanduser(v) for k, v in user_paths_dict.items()}
