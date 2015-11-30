from network import NetworkX
from layer import *
import tensorflow as tf
import yaml
import input_csv

config = yaml.load(file("config.yaml"))


berlin_net = NetworkX("Berlin",
                      [ConvolutionLayer(6, 6, 1, 1, 12),
                       PoolingLayer(2, 2, 2, 2),
                       ConvolutionLayer(6, 6, 1, 1, 12),
                       PoolingLayer(2, 2, 2, 2),
                       ConvolutionLayer(6, 6, 1, 1, 12),
                       PoolingLayer(2, 2, 2, 2),
                       FullyConnectedLayer(1024),
                       SoftmaxLinearLayer(2)])


training_set = input_csv.CSVInput(config['TRAINING_DATA'], config['INPUT_SHAPE'], config['OUTPUT_SHAPE'][0])
test_set = input_csv.CSVInput(config['TEST_DATA'], config['INPUT_SHAPE'], config['OUTPUT_SHAPE'][0])

berlin_net.add_input(training_set, test_set)
berlin_net.build()
berlin_net.set_cost()
berlin_net.set_optimizer(config['LEARNING_RATE'])
berlin_net.set_accuracy()
berlin_net.set_log_path(config['LOG_PATH'])
berlin_net.set_snapshot_path(config['SNAPSHOT_PATH'])

berlin_net.run(config['BATCH_SIZE'], config['TRAINING_ITERS'])

