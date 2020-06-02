import sys
import pipeline as pp
import json
import yaml
import os

if __name__ == '__main__':
    print(sys.argv)
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
        setting = yaml.load(file, Loader=yaml.FullLoader)
        print(setting)

        model = setting['model'].lower()
        mode = setting['mode'].lower()
        dataset = setting['dataset'].lower()

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
        test_path = None

    pp.single_prediction(test_path, model, setting, text)
