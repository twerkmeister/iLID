# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Modified by Thomas Werkmeister (thomas@twerkmeister.com)
# The original file can be found at:
# https://github.com/tensorflow/tensorflow/blob/1c57936/tensorflow/models/image/cifar10/cifar10.py

"""Builds the network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

import tensorflow.python.platform
from six.moves import urllib
import tensorflow as tf

import mel_spectrogram_input as image_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/vegeboy/workspace/uni/iLID-Data/output',
                           """Path to the data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_HEIGHT = image_input.IMAGE_HEIGHT
IMAGE_WIDTH = image_input.IMAGE_WIDTH
NUM_CLASSES = image_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = image_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = image_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable(name, shape, initializer):
  """Helper to create a Variable

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  return image_input.distorted_inputs(data_dir=FLAGS.data_dir,
                                        batch_size=FLAGS.batch_size)


def inputs(eval_data):
  """Construct input for evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  return image_input.inputs(eval_data=eval_data, data_dir=FLAGS.data_dir,
                              batch_size=FLAGS.batch_size)


def inference(images):
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[6, 6, 1, 12],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable('biases', [12], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)
  print("conv1:", conv1.get_shape())
  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[6, 6, 12, 12],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable('biases', [12], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # conf 3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[6, 6, 12, 12],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable('biases', [12], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # norm3
  norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm3')
  # pool3
  pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool3')

  # local4
  with tf.variable_scope('local4') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = 1
    for d in pool3.get_shape()[1:].as_list():
      dim *= d
    reshape = tf.reshape(pool3, [FLAGS.batch_size, dim])
    print(dim)
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable('biases', [384], tf.constant_initializer(0.1))
    local4 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
    _activation_summary(local4)

  # local5
  with tf.variable_scope('local5') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable('biases', [192], tf.constant_initializer(0.1))
    local5 = tf.nn.relu_layer(local4, weights, biases, name=scope.name)
    _activation_summary(local5)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.nn.xw_plus_b(local5, weights, biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def labels_to_dense(labels):
  # Reshape the labels into a dense Tensor of
  # shape [batch_size, NUM_CLASSES].
  sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
  indices = tf.reshape(tf.range(FLAGS.batch_size), [FLAGS.batch_size, 1])
  concated = tf.concat(1, [indices, sparse_labels])
  dense_labels = tf.sparse_to_dense(concated,
                                    [FLAGS.batch_size, NUM_CLASSES],
                                    1.0, 0.0)
  return dense_labels

def loss(logits, dense_labels):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    dense_labels: converted labels in shape [batch_size, NUM_CLASSES]

  Returns:
    Loss tensor of type float.
  """

  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      logits, dense_labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op

def accuracy(logits, dense_labels):
  seen_german = tf.Variable(0, trainable=False)
  seen_english = tf.Variable(0, trainable=False)

  x = tf.nn.softmax(logits)

  correct_pred = tf.equal(tf.argmax(x, 1), tf.argmax(dense_labels, 1))

  german_samples = tf.equal(tf.constant(1, dtype="int64"), tf.argmax(dense_labels, 1))
  german_accuracy = tf.reduce_mean(tf.cast(tf.gather(correct_pred, tf.where(german_samples)), tf.float32))
  sum_german_samples = seen_german.assign_add(tf.reduce_sum(tf.cast(tf.gather(correct_pred, tf.where(german_samples)), tf.int32)))
  tf.scalar_summary("german_accuracy", german_accuracy)

  english_samples = tf.equal(tf.constant(0, dtype="int64"), tf.argmax(dense_labels, 1))
  english_accuracy = tf.reduce_mean(tf.cast(tf.gather(correct_pred, tf.where(english_samples)), tf.float32))
  sum_english_samples = seen_english.assign_add(tf.reduce_sum(tf.cast(tf.gather(correct_pred, tf.where(english_samples)), tf.int32)))
  tf.scalar_summary("english_accuracy", english_accuracy)

  german_predictions = tf.equal(tf.constant(1, dtype="int64"), tf.argmax(x, 1))
  german_predictions_count = tf.reduce_sum(tf.cast(german_predictions, tf.int32))

  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  tf.scalar_summary("accuracy", accuracy)
  return accuracy, english_accuracy, german_accuracy, german_predictions_count, sum_english_samples, sum_german_samples


def train(total_loss, global_step):
  """Train the model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
