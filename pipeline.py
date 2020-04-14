import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import yaml

from preprocessing import kor_preprocessing as kp
from model.cnn import BasicCnnClassification, CharCnnClassification

import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def naver_movie_pipeline(file_path: str, model: str, setting: dict):
    train_data = pd.read_csv(file_path, header=0, delimiter='\t')
    hyper_parameters = setting['hyper_parameters']

    train_text = list(train_data['document'])
    train_label = list(train_data['label'])

    mode = setting['mode']

    if mode == 'char':
        train_in, train_out, char_dict, label_dict = kp.char_base_vectorize(train_text, train_label)
    elif mode == 'token':
        train_in, train_out, char_dict, label_dict = kp.kor_tokenizing(train_text, train_label)
    else:
        return 0

    train_in = tf.keras.preprocessing.sequence.pad_sequences(train_in, padding='post', maxlen=setting['max_len'])

    train_in = np.array(train_in)
    train_out = np.array(train_out)

    hyper_parameters['embed_input'] = len(char_dict)
    hyper_parameters['out_dim'] = len(label_dict)

    if model == 'cnn-basic':
        train_model = BasicCnnClassification(hyper_parameters)
    elif model == 'char-cnn-basic':
        train_model = CharCnnClassification(hyper_parameters)
    else:
        print("not implemented model")
        return 0

    train_model.compile(optimizer=tf.keras.optimizers.Adam(lr=setting['learning_rate']),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=['acc'])

    # train_model.build(input_shape=(setting['batch_size']))
    # train_model.summary()

    # Need to run with various batch_sizes
    tf.config.experimental_run_functions_eagerly(True)

    history = train_model.fit(train_in, train_out, epochs=setting['epochs'], batch_size=setting['batch_size'],
                              validation_split=setting['test_ratio'], shuffle=True)

    save_route = os.path.abspath('trained/' + setting['name'])
    train_model.save_weights(save_route, overwrite=True)

    save_dicts(char_dict, label_dict, save_route)
    update_setting(setting, len(char_dict), len(label_dict))

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


def naver_movie_prediction(file_path: str, model: str, setting: dict, text: str = None):
    if text is None:
        test_data = pd.read_csv(file_path, header=0, delimiter='\t')

        train_text = test_data['document']
        try:
            train_label = test_data['label']
        except KeyError:
            train_label = None

    else:
        train_text = [text]
        train_label = None

    mode = setting['mode']
    load_route = os.path.abspath('trained/' + setting['name'])
    text_dict, label_dict = load_dicts(load_route)

    if mode == 'char':
        train_in, train_out, char_dict, label_dict = kp.char_base_vectorize(train_text, train_label, text_dict,
                                                                            label_dict, train=False)
    elif mode == 'token':
        train_in, train_out, char_dict, label_dict = kp.kor_tokenizing(train_text, train_label, text_dict,
                                                                       label_dict,
                                                                       train=False)
    else:
        return 0

    # train_in = np.array(train_in)
    train_in = tf.keras.preprocessing.sequence.pad_sequences(train_in, padding='post', maxlen=setting['max_len'])

    print(train_in)

    if train_out is not None:
        train_out = np.array(train_out)

    hyper_parameters = setting['hyper_parameters']

    if model == 'cnn-basic':
        prediction_model = BasicCnnClassification(hyper_parameters)
    elif model == 'char-cnn-basic':
        prediction_model = CharCnnClassification(hyper_parameters)
    else:
        print('Not implemented model')
        return 0

    prediction_model.load_weights(load_route)

    tf.config.experimental_run_functions_eagerly(True)

    prediction = prediction_model.predict(train_in)
    print(prediction)
    probs = prediction[0]
    print(probs)
    result = tf.argmax(prediction, 1)

    if train_out is not None:
        equality = tf.equal(result, train_out)
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        tf.print(accuracy)

    else:
        print('수치:', probs)
        if result[0] == 0:
            print("부정적!")
        else:
            print("긍정적!")
        tf.print(result)


def save_dicts(text_dict: dict, label_dict: dict, route: str):
    with open(route + '_text_dict', 'wb') as handle:
        pickle.dump(text_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(route + '_label_dict', 'wb') as handle:
        pickle.dump(label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dicts(route):
    with open(route + '_text_dict', 'rb') as handle:
        text_dict = pickle.load(handle)

    with open(route + '_label_dict', 'rb') as handle:
        label_dict = pickle.load(handle)

    return text_dict, label_dict


def update_setting(setting: dict, embed_input, outputs):
    setting['hyper_parameters']['embed_input'] = embed_input
    setting['hyper_parameters']['out_dim'] = outputs

    route = os.path.abspath('parameters/' + setting['name'] + '.yaml')

    with open(route, 'w') as outfile:
        yaml.dump(setting, outfile, default_flow_style=False)
