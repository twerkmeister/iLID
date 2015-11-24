import tensorflow as tf
import numpy as np
import yaml
from scipy.ndimage import imread
from vgg_m_net import VGG_M_2048_NET

config = yaml.load(file("config.yaml"))

# Create model
x = tf.placeholder(tf.types.float32, [None] + config["INPUT_SHAPE"])

prediction_input = np.empty([1] + config["INPUT_SHAPE"])
prediction_input[0:] = imread("/Users/therold/Google Drive/Uni/DeepAudio/Code/tensorflow/voxforge/train/spectrogram_7.png", mode="RGB")

net = VGG_M_2048_NET(x, config["OUTPUT_SHAPE"][0])
pred = tf.nn.softmax(net.get_last_output())

# Start Prediction
with tf.Session() as sess:

    saver = tf.train.Saver()
    saver.restore(sess, "/Users/therold/Google Drive/Uni/DeepAudio/Code/tensorflow/snapshots/VGG_M_2048_NET.tensormodel-896")

    prediction = sess.run(pred, feed_dict={x: prediction_input})
    label = tf.argmax(prediction, 1)

    print "Probabilities: ", prediction
    print "Label: ", label


