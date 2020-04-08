import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import kor_preprocessing as kp
from model.cnn import BasicCnnClassification

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def naver_movie_pipeline(file_path: str, model: str, setting: dict):
    train_data = pd.read_csv(file_path, header=0, delimiter='\t')
    hyper_parameters = setting['hyper_parameters']

    train_text = train_data['document']
    train_label = train_data['label']

    if model == 'char-cnn-basic':
        train_in, train_out, char_dict, label_dict = kp.char_base_vectorize(train_text, train_label)

        train_in = tf.keras.preprocessing.sequence.pad_sequences(train_in, padding='post')

        train_in = np.array(train_in)
        train_out = np.array(train_out)

        hyper_parameters['embed_input'] = len(char_dict)
        hyper_parameters['out_dim'] = len(label_dict)

        train_model = BasicCnnClassification(hyper_parameters)

        train_model.compile(optimizer=tf.keras.optimizers.Adam(lr=setting['learning_rate']),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=['acc'])

        train_model.build(input_shape=(setting['batch_size'], None))
        train_model.summary()

        history = train_model.fit(train_in, train_out, epochs=setting['epochs'], batch_size=setting['batch_size'],
                                  validation_split=setting['test_ratio'], shuffle=True)

        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
