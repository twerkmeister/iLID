import tensorflow as tf
import os
from layer import *
import util.timestamp

class Network(object):
    def __init__(self, name, input_shape, output_shape, hidden_layers):
        self.name = name
        self.layers = None
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.created_at = util.timestamp.current_timestamp()

        self.initialize_input()
        self.hidden_layers = hidden_layers

        self.build()

        self.training_set = None
        self.test_set = None
        self.cost = None
        self.optimizer = None
        self.log_path = None
        self.snapshot_path = None

    def initialize_input(self):
        self.x = tf.placeholder(tf.types.float32, [None] + self.input_shape)
        self.y = tf.placeholder(tf.types.float32, [None] + self.output_shape)
        self.append(InputLayer(self.x))

    def build(self):
        for hidden_layer in self.hidden_layers:
            self.append(hidden_layer)

        self.set_activation_summary()
        self.print_network()

    def append(self, layer):
        if self.layers:
            layer.connect(self.layers)
            self.layers = layer
        else:
            assert(layer.layer_type == "input")
            self.input_layer = layer
            self.layers = layer

        return self

    def print_network(self):
        def loop(layer):
            if layer.layer_type == "input":
                print layer.name, layer.out_shape.as_list()[1:]
            else:
                print layer.name, layer.in_shape.as_list()[1:], "->", layer.out_shape.as_list()[1:]
                loop(layer.previous_layer)
        loop(self.layers)

    def set_activation_summary(self):
        '''Log each layers activations and sparsity.'''
        for layer in self.hidden_layers:
            tf.histogram_summary(layer.name + '/activations', layer.output)
            tf.scalar_summary(layer.name + '/sparsity', tf.nn.zero_fraction(layer.output))


    def set_training_input(self, training_set, test_set):
        self.training_set = training_set
        self.test_set = test_set
        assert(training_set.input_shape == test_set.input_shape)
        assert(training_set.input_shape == self.input_shape)
        assert([training_set.num_labels] == self.output_shape)

    def set_cost(self, logits_cost_function = tf.nn.softmax_cross_entropy_with_logits):
        #Last Layer should be softmax_linear
        #assert(self.layers.layer_type == "softmax_linear")
        self.cost = tf.reduce_mean(logits_cost_function(self.layers.output, self.y))
        tf.scalar_summary("loss", self.cost)

    def set_optimizer(self, learning_rate, decay_steps, optimizer = tf.train.AdamOptimizer):
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps, 0.96, staircase=True)
        tf.scalar_summary("learning_rate", lr)
        self.optimizer = optimizer(learning_rate=lr).minimize(self.cost, global_step = global_step)

    def set_accuracy(self):
        correct_pred = tf.equal(tf.argmax(self.layers.output, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.types.float32))
        tf.scalar_summary("accuracy", self.accuracy)

    def make_path(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def make_path_name(self, path):
        return os.path.join(os.getcwd(), path, "{0}_{1}".format(self.name, self.created_at))

    def set_log_path(self, log_path):
        self.log_path = self.make_path_name(log_path)
        self.make_path(self.log_path)

    def set_snapshot_path(self, snapshot_path):
        self.snapshot_path = self.make_path_name(snapshot_path)
        self.make_path(self.snapshot_path)

    def predict(self, model_path, images):
        with tf.Session() as sess:
            pred = tf.nn.softmax(self.layers.output)
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            label_op = tf.argmax(pred, dimension=1)
            prediction, label = sess.run([pred, label_op], feed_dict={self.x: images})
            print "Probabilities: ", prediction
            print "Label: ", label
            print prediction.shape

        return prediction, label

    def train(self, batch_size, iterations, display_step = 100):
        init = tf.initialize_all_variables()
        self.merged_summary_op = tf.merge_all_summaries()

        with tf.Session() as sess:
            self.saver = tf.train.Saver()
            self.summary_writer = tf.train.SummaryWriter(self.log_path, sess.graph_def)
            sess.run(init)
            self.optimize(sess, batch_size, iterations, display_step)
            self.evaluate(sess, batch_size)

    def optimize(self, sess, batch_size, iterations, display_step):
            step = 0
            while step < iterations:
                batch_xs, batch_ys = self.training_set.next_batch(batch_size)
                tf.image_summary("Input Images", batch_xs, max_images=1000)
                sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys})

                if step % display_step == 0:
                    self.write_progress(sess, step, batch_size)

                step += 1

            print "Optimization Finished!"

    def write_progress(self, sess, step, batch_size):
        # batch_xs, batch_ys = self.test_set.next_batch(batch_size)
        batch_xs, batch_ys = self.test_set.read_all()
        print "calculating stats on {0} samples".format(batch_xs.shape[0])
        summary_str, acc, loss = sess.run([self.merged_summary_op, self.accuracy, self.cost], feed_dict={self.x: batch_xs, self.y: batch_ys})
        self.summary_writer.add_summary(summary_str, step)
        if self.snapshot_path:
            path = os.path.join(self.snapshot_path, self.name + ".tensormodel")
            self.saver.save(sess, path, global_step=step)
        print "Iter {0}, Loss= {1:.6f}, Training Accuracy= {2:.5f}".format(step, loss, acc)


    def evaluate(self, sess):
        #Accuracy
        batch.xs, batch.ys = self.test_set.read_all()
        print "Accuracy:", sess.run(eval_correct, feed_dict={self.x: batch_xs, self.y: batch_ys})
