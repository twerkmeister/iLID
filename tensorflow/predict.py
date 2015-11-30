import tensorflow as tf
import numpy as np
import yaml
from scipy.ndimage import imread
from network.instances import berlin_net as net
import networkinput

config = yaml.load(file("config.yaml"))

def predict(image_path, model_path):

    # Create model
    x = tf.placeholder(tf.types.float32, [None] + config["INPUT_SHAPE"])

    prediction_input = networkinput.read_png("image_path")

    pred = tf.nn.softmax(net.layers.output)

    # Start Prediction
    with tf.Session() as sess:

        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        prediction = sess.run(pred, feed_dict={x: prediction_input})
        label = tf.argmax(prediction, 1)

        print "Probabilities: ", prediction
        print "Label: ", label

    return prediction, label


if __name__ == "__main__":

    image = "/Users/therold/Google Drive/Uni/DeepAudio/Code/tensorflow/voxforge/train/spectrogram_7.png"
    model = "/Users/therold/Google Drive/Uni/DeepAudio/Code/tensorflow/snapshots/VGG_M_2048.tensormodel-1280"

    predict(image, model)