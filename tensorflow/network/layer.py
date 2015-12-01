import tensorflow as tf
import numpy as np

class Layer(object):
    layer_type = None

    def __init__(self):
        self.depth = "X"
        self.set_name()

    def set_name(self):
        self.name ="{0}_{1}".format(self.layer_type, self.depth)

    def connect(self, previous_layer):
        self.previous_layer = previous_layer
        self.depth = previous_layer.depth + 1 if previous_layer else 0
        self.input = previous_layer.output if previous_layer else None
        self.in_shape = previous_layer.out_shape if previous_layer else None

        if self.layer_type == "input":
            assert(self.input == None)
        else:
            assert(self.input != None)

        self.set_name()
        self.output = self._output()
        self.out_shape = self.output.get_shape()

        return self

    def _output(self):
        raise NotImplemented


class InputLayer(Layer):
    layer_type = "input"
    def __init__(self, input_placeholder):
        self.input_placeholder = input_placeholder
        super(InputLayer, self).__init__()
        self.connect(None)
    
    def _output(self):
        return self.input_placeholder


class HiddenLayer(Layer):
    def __init__(self):
        super(HiddenLayer, self).__init__()

    def create_weights(self, name, shape, initializer=tf.random_normal_initializer()):
        return tf.get_variable(name, shape, initializer=initializer)

    def create_bias(self, name, shape, initializer=tf.constant_initializer(0.1)):
        return self.create_weights(name, shape, initializer)

    def flatten_input(self):
        in_size = np.prod(self.in_shape.as_list()[1:])
        return in_size, tf.reshape(self.input, [-1, in_size])

class PoolingLayer(HiddenLayer):
    layer_type = "pool"
    def __init__(self, kx, ky, sx=1, sy=1, padding="SAME", pooling_function=tf.nn.max_pool):
        super(PoolingLayer, self).__init__()
        self.kx = kx
        self.ky = ky
        self.sx = sx
        self.sy = sy
        self.padding = padding
        self.pooling_function = pooling_function

    def _output(self):
        return self.pooling_function(self.input, 
                              ksize=[1, self.kx, self.ky, 1], 
                              strides=[1, self.sx, self.sy, 1], 
                              padding=self.padding)

class ConvolutionLayer(HiddenLayer):
    layer_type="conv"
    
    def __init__(self, kx, ky, sx, sy, out_channels, padding="SAME", activation_function = tf.nn.relu):
        super(ConvolutionLayer, self).__init__()
        self.kx = kx
        self.ky = ky
        self.sx = sx
        self.sy = sy
        self.padding = padding
        self.out_channels = out_channels
        self.activation_function = activation_function

    def _output(self):
        with tf.variable_scope(self.name) as scope:
            in_channels = self.in_shape[3].value
            
            kernel = self.create_weights("weights", [self.kx, self.ky, in_channels, self.out_channels])
            bias = self.create_bias("bias", [self.out_channels])

            return self.activation_function(
                     tf.nn.bias_add(
                        tf.nn.conv2d(self.input, 
                                     kernel, 
                                     strides=[1, self.sx, self.sy, 1], 
                                     padding='SAME'),
                        bias),
                    name=scope.name)

class FullyConnectedLayer(HiddenLayer):
    layer_type = "fc"

    def __init__(self, out_size, activation_function = tf.nn.relu):
        super(FullyConnectedLayer, self).__init__()
        self.out_size = out_size
        self.activation_function = activation_function

    def _output(self):
        with tf.variable_scope(self.name) as scope:
            in_size, input_flat = self.flatten_input()
            weights = self.create_weights("weights", [in_size, self.out_size])
            bias = self.create_bias("bias", [self.out_size])
            
            return self.activation_function(tf.matmul(input_flat, weights) + bias, name=scope.name)

class DropoutLayer(HiddenLayer):
    layer_type = "dropout"

    def __init__(self, dropout_rate):
        super(DropoutLayer, self).__init__()
        self.dropout_rate = dropout_rate

    def _output(self):
        return tf.nn.dropout(self.input, self.dropout_rate)
