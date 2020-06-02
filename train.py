import sys
import pipeline as pp
import json
import yaml
from yaml.scanner import ScannerError
import os

if __name__ == '__main__':
    '''
        run with config file
    '''

    # 설정 가능한 model / mode 리스트
    model_list = ['BasicCnnClassification', 'CharCnnClassification']
    mode_list = ['char', 'token']
    dataset_list = ['nsmc', 'imdb']

    if len(sys.argv) == 1:
        sys.exit('No config file provided')

    try:
        path = os.path.abspath('parameters/' + sys.argv[1])

        with open(path, encoding='UTF-8') as file:
            setting = yaml.load(file, Loader=yaml.FullLoader)
            print(setting)

        file.close()

    except FileNotFoundError:
        sys.exit('Not yaml file found')

    except ScannerError:
        sys.exit('Not Correct yaml file')

    try:
        model = setting['model'].lower()
        mode = setting['mode'].lower()
        dataset = setting['dataset'].lower()

    except KeyError:
        sys.exit("Can not find Model, mode, or dataset from json")

    if model not in [x.lower() for x in model_list]:
        print("Wrong or Not Contained Model, Please Choose in ", model_list)
        sys.exit()

    if mode not in mode_list:
        print("Wrong or Not Contained Mode, Please Choose in ", mode_list)
        sys.exit()

    if dataset not in dataset_list:
        print("Wrong or Not Contained dataset, Please Choose in ", mode_list)
        sys.exit()

    try:
        hyper_params = setting['hyper_parameters']

    # hyper parameter 없으면, default 값에서 가져옴
    except KeyError:
        if model == 'BasicCnnClassification' or model == 'CharCnnClassification':
            path = os.path.abspath('parameters/default_parameter.yaml')
            with open(path, encoding='UTF-8') as f:
                default_setting = yaml.load(file, Loader=yaml.FullLoader)

            hyper_params = default_setting['hyper_parameters']
        else:
            print('not yet implemented model')
            sys.exit()
    # 파라미터 없으니 채움
    setting['hyper_parameters'] = hyper_params

    # dataset 에 맞는 path 설정
    path = os.path.abspath('datasets/dataset_setting.yaml')
    with open(path, encoding='UTF-8') as file:
        dataset_path = yaml.load(file, Loader=yaml.FullLoader)

    if dataset == 'nsmc':
        train_path = dataset_path['nsmc']['train']
        test_path = dataset_path['nsmc']['test']

    elif dataset == 'imdb':
        train_path = dataset_path['imdb']['train']
        test_path = dataset_path['imdb']['test']

    else:
        train_path = None

    pp.train_pipeline(train_path, model, setting)
