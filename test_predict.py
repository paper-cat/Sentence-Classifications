# -*-coding:utf-8-*-

import sys
import pipeline as pp
import yaml
import os


def test_predict():
    sys.argv = ['predict.py', 'nsmc_tokens', '존잼!']
    # sys.argv = ['predict.py', 'imdb_token', 'Great', 'Movie']

    path = os.path.abspath('trained/' + sys.argv[1])

    # argument 이후의 str 을 하나의 문장으로
    if len(sys.argv) > 2:
        text = ''
        for i in range(len(sys.argv) - 2):
            if text == '':
                text += sys.argv[i + 2]
            else:
                text += ' '
                text += sys.argv[i + 2]

    else:
        text = None

    print(text)

    with open(path + '/' + sys.argv[1] + '_setting.yaml', encoding='UTF-8') as file:
        setting = dict(yaml.load(file, Loader=yaml.FullLoader))
        print(setting)

    model = setting['model'].lower()
    dataset = setting['dataset'].lower()

    path = os.path.abspath('datasets/dataset_setting.yaml')
    with open(path, encoding='UTF-8') as file:
        dataset_path = yaml.load(file, Loader=yaml.FullLoader)

    if dataset == 'nsmc':
        test_path = dataset_path['nsmc']['test']
    elif dataset == 'imdb':
        test_path = dataset_path['imdb']['test']

    else:
        test_path = None

    result = pp.single_prediction(test_path, model, setting, text)

    assert result == 1


if __name__ == '__main__':
    test_predict()