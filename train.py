import sys
import pipeline as pp
import json
import yaml
from yaml.scanner import ScannerError
import os

if __name__ == '__main__':
    '''
        run with 1. train file path 2. train mode, 3. model-selection
    '''

    model_list = ['char-cnn-basic', 'char-cnn-custom', 'word-cnn']
    mode_list = ['nsmc']

    if len(sys.argv) == 1:
        sys.exit('No config file provided')

    try:
        path = os.path.abspath('parameters/' + sys.argv[1])

        with open(path) as file:
            setting = yaml.load(file, Loader=yaml.FullLoader)
            print(setting)

    except FileNotFoundError:
        sys.exit('Not yaml file found')

    except ScannerError:
        sys.exit('Not Correct yaml file')

    try:
        model = setting['model'].lower()
        mode = setting['mode'].lower()
        train_file_path = setting['train_file_path'].lower()

    except KeyError:
        sys.exit("Can not find Model, mode, or train file path from json")

    if model not in model_list:
        print("Wrong or Not Contained Model, Please Choose in ", model_list)
        sys.exit()

    if mode not in mode_list:
        print("Wrong or Not Contained Mode, Please Choose in ", mode_list)
        sys.exit()

    try:
        hyper_params = setting['hyper_parameters']
    except KeyError:
        if mode == 'nsmc':
            path = os.path.abspath('parameters/nsmc_default.yaml')
            with open(path) as f:
                default_setting = yaml.load(file, Loader=yaml.FullLoader)

            hyper_params = default_setting['hyper_parameters']
        else:
            print('not yet implemented')
            sys.exit()
    setting['hyper_parameters'] = hyper_params

    print('{:20}'.format('Train mode'), ': ', mode)
    print('{:20}'.format('Train Data File'), ': ', train_file_path)

    if mode == 'nsmc':
        pp.naver_movie_pipeline(train_file_path, model, setting)
    else:
        print('Not Implemented')
