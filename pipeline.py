import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import yaml
import logging
import json
import tensorflow_datasets as tfds

from preprocessing import text_preprocessing as tp
from model.cnn import CnnYoonKim, CharCnnClassification

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def train_pipeline(file_path: str, model: str, setting: dict):
    hyper_parameters = setting['hyper_parameters']
    dataset = setting['dataset']

    print('Preprocessing Text Data...')
    if dataset == 'nsmc':
        train_data = pd.read_csv(file_path, header=0, delimiter='\t')
        train_text = list(train_data['document'])
        train_label = list(train_data['label'])
    elif dataset == 'imdb':
        train_ds = tfds.load('imdb_reviews', split='train', shuffle_files=True, data_dir=file_path)
        train_text = [x['text'].numpy().decode() for x in train_ds]
        train_label = [int(x['label']) for x in train_ds]
    else:
        print('not yet implemented dataset')
        return 0

    mode = setting['mode']
    if mode == 'char':
        train_in, train_out, char_dict, label_dict = tp.char_base_vectorize(train_text, train_label)
    elif mode == 'token':
        if dataset == 'nsmc':
            train_in, train_out, char_dict, label_dict = tp.tokenizing(train_text, train_label, language='kor')
        elif dataset == 'imdb':
            train_in, train_out, char_dict, label_dict = tp.tokenizing(train_text, train_label, language='eng')
        else:
            print('not implemented dataset came in')
            return 0
    else:
        return 0

    train_in = tf.keras.preprocessing.sequence.pad_sequences(train_in, padding='post', maxlen=setting['max_len'])

    train_in = np.array(train_in)
    train_out = np.array(train_out)

    hyper_parameters['embed_input'] = len(char_dict)
    hyper_parameters['out_dim'] = len(label_dict)

    if model == 'CnnYoonKim'.lower():
        train_model = CnnYoonKim(hyper_parameters)
    elif model == 'CharCnnClassification'.lower():
        train_model = CharCnnClassification(hyper_parameters)
    else:
        print("not implemented model")
        return 0

    train_model.compile(optimizer=tf.keras.optimizers.Adam(lr=hyper_parameters['learning_rate']),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=['acc'])

    # Need to run with various batch_sizes
    tf.config.experimental_run_functions_eagerly(True)

    # train_model.build(input_shape=(hyper_parameters['batch_size'], len(train_in[0])))
    # train_model.summary()

    history = train_model.fit(train_in, train_out,
                              epochs=hyper_parameters['epochs'],
                              batch_size=hyper_parameters['batch_size'],
                              validation_split=setting['test_ratio'],
                              shuffle=True)

    log_text = dict({'accuracy': history.history['acc'],
                     'loss': history.history['loss']})

    if setting['test_ratio'] > 0:
        log_text['val_accuracy'] = history.history['val_acc']
        log_text['val_loss'] = history.history['val_loss']

    save_route = os.path.abspath('trained/' + setting['name'])
    train_model.save_weights(save_route + '/model', overwrite=True)

    save_dicts(char_dict, label_dict, save_route + '/')
    update_setting(setting, len(char_dict), len(label_dict), save_route)

    # make log file
    make_log_file(log_text, save_route)

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    if setting['test_ratio'] > 0:
        plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    if setting['test_ratio'] > 0:
        plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def single_prediction(file_path: str, model: str, setting: dict, text: str = None):
    dataset = setting['dataset']
    mode = setting['mode']

    if dataset == 'nsmc':
        lang = 'kor'
    elif dataset == 'imdb':
        lang = 'eng'
    else:
        return 0

    if text is None:
        # file testing
        if dataset == 'nsmc':
            test_data = pd.read_csv(file_path, header=0, delimiter='\t')
            test_text = test_data['document']
            try:
                test_label = test_data['label']
            except KeyError:
                test_label = None

        elif dataset == 'imdb':
            test_ds = tfds.load('imdb_reviews', split='test', shuffle_files=False, data_dir=file_path)
            test_text = [str(x['text']) for x in test_ds]
            test_label = [int(x['label']) for x in test_ds]
        else:
            print('Other than nsmc, imdb is not implemented yet')
            return 0
    else:
        test_text = [text]
        test_label = None

    load_route = os.path.abspath('trained/' + setting['name'])
    text_dict, label_dict = load_dicts(load_route)

    if mode == 'char':
        train_in, train_out, char_dict, label_dict = tp.char_base_vectorize(test_text,
                                                                            test_label,
                                                                            text_dict,
                                                                            label_dict,
                                                                            train=False)
    elif mode == 'token':
        train_in, train_out, char_dict, label_dict = tp.tokenizing(test_text,
                                                                   test_label,
                                                                   text_dict,
                                                                   label_dict,
                                                                   train=False,
                                                                   language=lang)
    else:
        return 0

    train_in = tf.keras.preprocessing.sequence.pad_sequences(train_in, padding='post', maxlen=setting['max_len'])

    if train_out is not None:
        train_out = np.array(train_out)

    hyper_parameters = setting['hyper_parameters']

    if model == 'CnnYoonKim'.lower():
        prediction_model = CnnYoonKim(hyper_parameters)
    elif model == 'CharCnnClassification'.lower():
        prediction_model = CharCnnClassification(hyper_parameters)
    else:
        print('Not implemented model')
        return 0

    prediction_model.load_weights(load_route + '/model')

    tf.config.experimental_run_functions_eagerly(True)

    prediction = prediction_model.predict(train_in)
    probs = prediction[0]

    if text is None:
        print(prediction)
        print(probs)

    result = tf.argmax(prediction, 1)

    if train_out is not None:
        equality = tf.equal(result, train_out)
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        tf.print('Test file Accuracy : ', accuracy)

    elif dataset in ['nsmc', 'imdb']:
        print('Probability:', probs[np.argmax(result)])
        if result[0] == 0:
            print("Negative!")
            return 0
        else:
            print("Positive!")
            return 1
    else:
        return None


def save_dicts(text_dict: dict, label_dict: dict, route: str):
    with open(route + 'text_dict', 'wb') as handle:
        pickle.dump(text_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(route + 'label_dict', 'wb') as handle:
        pickle.dump(label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dicts(route):
    with open(route + '/text_dict', 'rb') as handle:
        text_dict = pickle.load(handle)

    with open(route + '/label_dict', 'rb') as handle:
        label_dict = pickle.load(handle)

    return text_dict, label_dict


def update_setting(setting: dict, embed_input, outputs, save_route):
    setting['hyper_parameters']['embed_input'] = embed_input
    setting['hyper_parameters']['out_dim'] = outputs

    route = os.path.abspath(save_route + '/' + setting['name'] + '_setting.yaml')

    with open(route, 'w') as outfile:
        yaml.dump(setting, outfile, default_flow_style=False)


def make_log_file(logs: dict, route: str):
    with open(route + '/log.txt', 'w') as file:
        file.write(json.dumps(logs))
