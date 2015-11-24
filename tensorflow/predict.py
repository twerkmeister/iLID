import tensorflow as tf
import numpy as np
import yaml
from scipy.ndimage import imread
from vgg_m_net import VGG_M_2048_NET as Network

config = yaml.load(file("config.yaml"))

def predict(image_path, model_path):

    # Create model
    x = tf.placeholder(tf.types.float32, [None] + config["INPUT_SHAPE"])

    prediction_input = np.empty([1] + config["INPUT_SHAPE"])
    prediction_input[0:] = imread(image_path, mode="RGB")

    net = Network(x, config["OUTPUT_SHAPE"][0])
    pred = tf.nn.softmax(net.get_last_output())

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