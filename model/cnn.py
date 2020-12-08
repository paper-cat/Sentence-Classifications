import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Dropout, GlobalMaxPool1D, MaxPooling1D, Flatten


# https://arxiv.org/pdf/1408.5882v2.pdf Yoon Kim
class CnnYoonKim(tf.keras.Model):
    def __init__(self, args: dict):
        super(CnnYoonKim, self).__init__()

        self.embed_input = args['embed_input']
        self.embed_out = args['embedding']
        self.cnn_filters = args['cnn_filters']
        self.dropout_rate = args['dropout']
        self.hidden_units = args['hidden_units']
        self.out_dim = args['out_dim']

        self.embedding = Embedding(self.embed_input, self.embed_out, mask_zero=True)
        self.convolutions = [
            Conv1D(self.cnn_filters, x, activation='relu', kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.),
                   padding='same') for x in args['kernel_size']]
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
class CharCnnClassification(tf.keras.Model):
    def __init__(self, args: dict):
        super(CharCnnClassification, self).__init__()

        self.embed_input = args['embed_input']
        self.embed_out = args['embedding']
        self.cnn_filters = args['cnn_filters']
        self.cnn_kernels = args['kernel_size']
        self.max_pools_units = args['max_pools']
        self.dropout_rate = args['dropout']
        self.hidden_units = args['hidden_units']
        self.out_dim = args['out_dim']

        self.embedding = Embedding(self.embed_input, self.embed_out, mask_zero=True)

        self.conv_list = []
        for i in range(min(len(self.cnn_filters), len(self.cnn_kernels))):
            self.conv_list.append(
                Conv1D(self.cnn_filters[i], kernel_size=self.cnn_kernels[i], activation='relu', padding='same',
                       kernel_initializer=tf.random_normal_initializer(0.0, 0.02)))

        self.max_pools = []
        for i in range(len(self.max_pools_units)):
            if self.max_pools_units[i] == 0 or self.max_pools_units is None:
                self.max_pools.append(None)
            else:
                self.max_pools.append(MaxPooling1D(self.max_pools_units[i], padding='same'))

        self.flat = Flatten()
        self.dropout1 = Dropout(self.dropout_rate)
        # self.dropout2 = Dropout(self.dropout_rate)

        self.fcn1 = Dense(self.hidden_units, activation='relu', activity_regularizer='l2')
        # self.fcn2 = Dense(self.hidden_units, 'relu')
        self.out_fcn = Dense(self.out_dim, 'softmax')

    @tf.function
    def call(self, x):
        x = self.embedding(x)

        for i in range(len(self.conv_list)):
            x = self.conv_list[i](x)
            if self.max_pools[i] is not None:
                x = self.max_pools[i](x)

        x = self.flat(x)
        x = self.fcn1(x)
        x = self.dropout1(x)
        # x = self.fcn2(x)
        # x = self.dropout2(x)
        x = self.out_fcn(x)

        return x


if __name__ == '__main__':
    pass
