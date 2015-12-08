import tensorflow as tf
import numpy as np
import yaml
from scipy.ndimage import imread
from network.instances.berlinnet_unnormal import net
import networkinput
import argparse

config = yaml.load(file("config.yaml"))

def evaluate(model_path):
    training_set = networkinput.CSVInput(config['TRAINING_DATA'], config['INPUT_SHAPE'], config['OUTPUT_SHAPE'][0], mode="L")
    test_set = networkinput.CSVInput(config['TEST_DATA'], config['INPUT_SHAPE'], config['OUTPUT_SHAPE'][0], mode="L")

    net.set_training_input(training_set, test_set)
    net.load_and_evaluate(model_path)

if __name__ == "__main__":  

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_path', required=True, help='Path to saved tensorflow model')

    args = parser.parse_args()

    evaluate(args.model_path)