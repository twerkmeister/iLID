from network import *
import tensorflow as tf

labels = 2
net = Network("VGG_M_2048",
                    [224, 224, 3],
                    [labels],
                    [ConvolutionLayer(7, 7, 2, 2, 96),
                     PoolingLayer(3, 3, 2, 2),
                     ConvolutionLayer(5, 5, 2, 2, 256),
                     PoolingLayer(3, 3, 2, 2),
                     ConvolutionLayer(3, 3, 1, 1, 512),
                     ConvolutionLayer(3, 3, 1, 1, 512),
                     ConvolutionLayer(3, 3, 1, 1, 512),
                     PoolingLayer(3, 3, 2, 2),
                     FullyConnectedLayer(4096),
                     DropoutLayer(0.5),
                     FullyConnectedLayer(2048),
                     DropoutLayer(0.5),
                     FullyConnectedLayer(2048),
                     FullyConnectedLayer(labels, activation_function=tf.identity)])