import tensorflow as tf
from input_csv import CSVInput
from vgg_m_net import VGG_M_2048_NET

# Parameters
LEARNING_RATE = 0.001
TRAINING_ITERS = 100000
BATCH_SIZE = 128
DISPLAY_STEP = 10

# Network Parameters
INPUT_SHAPE = [224, 224, 3]  # Input Image shape
NUM_CLASSES = 1  # Total classes

# Create model
x = tf.placeholder(tf.types.float32, [None] + INPUT_SHAPE)
y = tf.placeholder(tf.types.float32, [None, NUM_CLASSES])

net = VGG_M_2048_NET(x, NUM_CLASSES)
pred = net.get_last_output()

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.types.float32))

# Train
init = tf.initialize_all_variables()
train_data = CSVInput("/Users/therold/Google Drive/Uni/DeepAudio/Code/tensorflow/train.csv", BATCH_SIZE, INPUT_SHAPE)

with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * BATCH_SIZE < TRAINING_ITERS:

        batch_xs, batch_ys = train_data.next_batch()
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

        if step % DISPLAY_STEP == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            print "Iter " + str(step * BATCH_SIZE) + ", Loss= " + "{:.6f}".format(
                loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1

    print "Optimization Finished!"

    # Accuracy on 256 mnist test images
    print "Accuracy:", sess.run(accuracy,
                                feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
