import tensorflow as tf


class Network(object):
    def __init__(self):
        self.name = "Unnamed"
        self.layers = []

        self.input_shape = None # [20, 20, 3]  w x h x channels
        self.output_shape = None # [2] one hot vector for classes

    def build_net(self, input):
        raise Exception("Need to be implemented in subclass!")

    def create_variable(self, name, shape):
        return tf.get_variable(name, shape, initializer=tf.random_normal_initializer())

    def add_layer(self, name, layer):
        self.layers.append((name, layer))

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t, _ in self.layers) + 1
        return '%s_%d' % (prefix, id)

    def get_last_output(self):
        return self.layers[-1][1]

    def input(self, input):
        self.add_layer("input", input)
        return self

    def conv(self, kx, ky, sx, sy, in_size, out_size, name=None):
        name = name or self.get_unique_name("conv")

        with tf.variable_scope(name) as scope:
            input = self.get_last_output()
            kernel = self.create_variable("weights", [kx, ky, in_size, out_size])
            bias = self.create_variable("bias", [out_size])

            conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, kernel, strides=[1, sx, sy, 1], padding='SAME'), bias),
                              name=scope.name)
            self.add_layer(name, conv)

        return self

    def pool(self, kx, ky, sx=1, sy=1, name=None):
        name = name or self.get_unique_name("pool")

        input = self.get_last_output()
        pool = tf.nn.max_pool(input, ksize=[1, kx, ky, 1], strides=[1, sx, sy, 1], padding='SAME')
        self.add_layer(name, pool)

        return self

    def fc(self, in_size, out_size, name=None):
        name = name or self.get_unique_name("fc")

        with tf.variable_scope(name) as scope:
            input = self.get_last_output()
            weights = self.create_variable("weights", [in_size, out_size])
            bias = self.create_variable("bias", [out_size])

            input_flat = tf.reshape(input, [-1, weights.get_shape().as_list()[0]])
            fc = tf.nn.relu(tf.matmul(input_flat, weights) + bias, name=scope.name)

            self.add_layer(name, fc)

        return self

    def softmax_linear(self, in_size, out_size, name=None):
        name = name or self.get_unique_name("softmax_linear")

        with tf.variable_scope(name) as scope:
            input = self.get_last_output()
            weights = self.create_variable("weights", [in_size, out_size])
            bias = self.create_variable("bias", [out_size])

            softmax_linear = tf.nn.xw_plus_b(input, weights, bias, name=scope.name)

            self.add_layer(name, softmax_linear)

        return self

    def lrn(self, name=None):
        return self

    def dropout(self, dropout_rate, name=None):
        name = name or self.get_unique_name("dropout")

        input = self.get_last_output()
        dropout = tf.nn.dropout(input, dropout_rate)
        self.add_layer(name, dropout)

        return self

    def debug(self, name=None):

        # Make sure we have all the right params for the network.
        assert(self.input_shape)
        assert(self.output_shape)

        name = name or self.get_unique_name("debug")

        input = self.get_last_output()
        layer_name = filter(lambda (name, layer): layer == input, self.layers)[0][0]
        print "{0} : {1}".format(layer_name, input.get_shape())

        return self
