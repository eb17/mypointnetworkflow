"""
weighted_categorical_cross-entropy function
Implemented and taken form Mendi Barels stack-overflow post from 29. December 2019:
https://stackoverflow.com/questions/59520807/multi-class-weighted-loss-for-semantic-image-segmentation-in-keras-tensorflow
"""
import numpy as np
from keras import backend as K


def weighted_categorical_crossentropy(point_weights):
    def wcce(y_true, y_pred):
        k_weights = K.constant(point_weights)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * k_weights, axis=-1)

    return wcce


if __name__ == '__main__':
    point_weights = [0.9, 0.05, 0.04]

    y_true = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1], [0.19, 0.8, 0.01]])
    loss = weighted_categorical_crossentropy(point_weights)
    print(loss(y_true.astype('float32'), y_pred.astype('float32')).numpy())
