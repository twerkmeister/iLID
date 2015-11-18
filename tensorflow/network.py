import tensorflow as tf


class Network(object):
    def __init__(self, input):
	self.layers = []
	self.add_layer("input", input)

    def create_variable(self, name, shape):
	return tf.get_variable(name, shape, initializer=tf.random_normal_initializer())

    def add_layer(self, name, layer):
	self.layers.append((name, layer))

    def get_unique_name(self, prefix):
	id = sum(t.startswith(prefix) for t, _ in self.layers) + 1
	return '%s_%d' % (prefix, id)

    def get_last_output(self):
	return self.layers[-1][1]

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

    def lrn(self, name=None):
	return self

    def dropout(self, dropout_rate, name=None):
	name = name or self.get_unique_name("dropout")

	input = self.get_last_output()
	dropout = tf.nn.dropout(input, dropout_rate)
	self.add_layer(name, dropout)

	return self

    def debug(self, name=None):
	name = name or self.get_unique_name("debug")

	input = self.get_last_output()
	layer_name = filter(lambda (name, layer): layer == input, self.layers)[0][0]
	print "{0} : {1}".format(layer_name, input.get_shape())

	return self
