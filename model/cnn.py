import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv1D


class CnnClassification(tf.keras.Model):
    def __init__(self, args: dict):
        super(CnnClassification, self).__init__()
        self.convolution = Conv1D(int(args["units"]), 3, 1)

    def call(self, inputs):
        x = self.convolution(inputs)

        return x


if __name__ == '__main__':
    pass
