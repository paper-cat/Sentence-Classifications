import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Dropout, GlobalMaxPool1D


# https://arxiv.org/pdf/1408.5882v2.pdf Yoon Kim
class BasicCnnClassification(tf.keras.Model):
    def __init__(self, args: dict):
        super(BasicCnnClassification, self).__init__()

        self.embed_input = args['embed_input']
        self.embed_out = args['embedding']
        self.cnn_filters = args['cnn_filters']
        self.dropout_rate = args['dropout']
        self.hidden_units = args['hidden_units']
        self.out_dim = args['out_dim']

        self.embedding = Embedding(self.embed_input, self.embed_out, mask_zero=True)
        self.convolutions = [
            Conv1D(self.cnn_filters, x, activation='relu', kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))
            for x in args['kernel_size']]
        self.gmp = GlobalMaxPool1D()
        self.dropout = Dropout(self.dropout_rate)
        self.fcn = Dense(self.hidden_units, 'relu')
        self.out_fcn = Dense(self.out_dim, 'softmax')

    @tf.function
    def call(self, x):
        x = self.embedding(x)
        x = tf.concat([self.gmp(c(x)) for c in self.convolutions], axis=-1)
        x = self.dropout(x)
        x = self.fcn(x)
        x = self.dropout(x)
        x = self.out_fcn(x)

        return x


# https://arxiv.org/pdf/1509.01626v3.pdf

if __name__ == '__main__':
    pass
