from network import *
import tensorflow as tf


def generate(input_shape, num_labels):
    return Network("Berlin2",
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
                    FullyConnectedLayer(2048),
                    FullyConnectedLayer(1024),
                    FullyConnectedLayer(num_labels, activation_function=tf.identity)])
