from network import *

labels = 2
berlin_net = Network("Berlin",
                      [39, 600, 1],
                      [labels],
                      [ConvolutionLayer(6, 6, 1, 1, 12),
                       PoolingLayer(2, 2, 2, 2),
                       ConvolutionLayer(6, 6, 1, 1, 12),
                       PoolingLayer(2, 2, 2, 2),
                       ConvolutionLayer(6, 6, 1, 1, 12),
                       PoolingLayer(2, 2, 2, 2),
                       FullyConnectedLayer(1024),
                       SoftmaxLinearLayer(labels)])