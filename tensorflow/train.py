import tensorflow as tf
import yaml
import os
from input_csv import CSVInput
#from vgg_m_net import VGG_M_2048_NET as Network
from berlin_net import BERLIN_NET as Network

config = yaml.load(file("config.yaml"))

# Create model
net = Network()
x = tf.placeholder(tf.types.float32, [None] + net.input_shape)
y = tf.placeholder(tf.types.float32, [None] + net.output_shape)

# Learning operation
logits = net.build_net(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer(learning_rate=config["LEARNING_RATE"]).minimize(cost)
tf.scalar_summary("loss", cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.types.float32))
tf.scalar_summary("accuracy", accuracy)

# Datasets
init = tf.initialize_all_variables()
train_data = CSVInput(config["TRAINING_DATA"], config["BATCH_SIZE"], net.input_shape, net.output_shape)
test_data = CSVInput(config["TEST_DATA"], config["BATCH_SIZE"], net.input_shape, net.output_shape)

# Summary for Tensorboard
merged_summary_op = tf.merge_all_summaries()

# Start Training
with tf.Session() as sess:

    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(config["LOG_PATH"], sess.graph_def)

    sess.run(init)
    step = 1

    while step * config["BATCH_SIZE"] < config["TRAINING_ITERS"]:

        # Learn weights
        batch_xs, batch_ys = train_data.next_batch()
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

        # Write logs and display intermediary result
        if step % config["DISPLAY_STEP"] == 0:
            batch_xs, batch_ys = test_data.next_batch()

            summary_str, acc, loss = sess.run([merged_summary_op, accuracy, cost], feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, step)

            iteration = str(step * config["BATCH_SIZE"])
            path = "{0}.tensormodel".format(os.path.join(config["SNAPSHOT_PATH"], net.name))
            saver.save(sess, path, global_step=step * config["BATCH_SIZE"])
            print "Iter {0}, Loss= {1:.6f}, Training Accuracy= {2:.5f}".format(iteration, loss, acc)

        step += 1

    print "Optimization Finished!"

    #Accuracy
    batch_xs, batch_ys = test_data.next_batch()
    print "Accuracy:", sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})


