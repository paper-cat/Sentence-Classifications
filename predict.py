import sys
import pipeline as pp
import json
import yaml
import os

if __name__ == '__main__':
    print(sys.argv)
    path = os.path.abspath('parameters/' + sys.argv[1])

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

    with open(path, encoding='UTF-8') as file:
        setting = yaml.load(file, Loader=yaml.FullLoader)
        print(setting)

        model = setting['model'].lower()
        mode = setting['mode'].lower()
        dataset = setting['dataset'].lower()
        test_file_path = setting['test_file_path'].lower()

        if dataset == 'nsmc':
            pp.naver_movie_prediction(test_file_path, model, setting, text)
