import tensorflow as tf
import numpy as np
import yaml
from scipy.ndimage import imread
from network.instances.berlinnet import net
import networkinput
import argparse

config = yaml.load(file("config.yaml"))

def predict(image_path, model_path):
    image = networkinput.read_png(image_path, "L")
    net.predict(model_path, np.expand_dims(image, axis=0))

if __name__ == "__main__":  

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='image_path', help='Path to image file', required=True)
    parser.add_argument('--model', dest='model_path', required=True, help='Path to saved tensorflow model')

    args = parser.parse_args()

    predict(args.image_path, args.model_path)