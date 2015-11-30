from network import *

berlin_net = Network("Berlin",
                      [ConvolutionLayer(6, 6, 1, 1, 12),
                       PoolingLayer(2, 2, 2, 2),
                       ConvolutionLayer(6, 6, 1, 1, 12),
                       PoolingLayer(2, 2, 2, 2),
                       ConvolutionLayer(6, 6, 1, 1, 12),
                       PoolingLayer(2, 2, 2, 2),
                       FullyConnectedLayer(1024),
                       SoftmaxLinearLayer(2)])