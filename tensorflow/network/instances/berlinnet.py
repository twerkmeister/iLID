from network import *
import tensorflow as tf
import numpy as np


def generate(input_shape, num_labels):
    return Network("Berlin",
                   input_shape,
                   [num_labels],
                   [ConvolutionLayer(6, 6, 1, 1, 12),
                    LocalResponseNormalizationLayer(),
                    PoolingLayer(2, 2, 2, 2),
                    ConvolutionLayer(6, 6, 1, 1, 12),
                    LocalResponseNormalizationLayer(),
                    PoolingLayer(2, 2, 2, 2),
                    ConvolutionLayer(6, 6, 1, 1, 12),
                    LocalResponseNormalizationLayer(),
                    PoolingLayer(2, 2, 2, 2),
                    FullyConnectedLayer(1024),
                    FullyConnectedLayer(num_labels, activation_function=tf.identity,
                      weights_initializer=tf.truncated_normal_initializer(stddev=1/float(1024)), 
                      bias_initializer=tf.constant_initializer(np.log(1/float(num_labels))))])
