import tensorflow as tf
from input_csv import CSVInput
from vgg_m_net import VGG_M_2048_NET

# Parameters
LEARNING_RATE = 0.001
TRAINING_ITERS = 100000
BATCH_SIZE = 128
DISPLAY_STEP = 10

# Network Parameters
INPUT_SHAPE = [224, 224, 3] # Input Image shape
NUM_CLASSES = 1 # Total classes

# Create model
x = tf.placeholder(tf.types.float32, [None] + INPUT_SHAPE)
y = tf.placeholder(tf.types.float32, [None, NUM_CLASSES])

net = VGG_M_2048_NET(x, NUM_CLASSES)
pred = net.get_last_output()

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
tf.scalar_summary("loss", cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.types.float32))
tf.scalar_summary("accuracy", accuracy)

# Train
init = tf.initialize_all_variables()
train_data = CSVInput("/Users/therold/Google Drive/Uni/DeepAudio/Code/tensorflow/train.csv", BATCH_SIZE, INPUT_SHAPE)
test_data = CSVInput("/Users/therold/Google Drive/Uni/DeepAudio/Code/tensorflow/train.csv", BATCH_SIZE, INPUT_SHAPE)

# Summary for Tensorboard
merged_summary_op = tf.merge_all_summaries()


# Start Training
with tf.Session() as sess:

    summary_writer = tf.train.SummaryWriter("logs", sess.graph_def)
    sess.run(init)
    step = 1

    while step * BATCH_SIZE < TRAINING_ITERS:

        batch_xs, batch_ys = train_data.next_batch()
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

        if step % DISPLAY_STEP == 0:
            batch_xs, batch_ys = test_data.next_batch()

            summary_str, acc, loss = sess.run([merged_summary_op, accuracy, cost], feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, step)

            print "Iter " + str(step * BATCH_SIZE) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1

    print "Optimization Finished!"

    # Finish summaries
    summary_str = sess.run(merged_summary_op)
    summary_writer.add_summary(summary_str, step)


    #Accuracy
    batch_xs, batch_ys = test_data.next_batch()
    print "Accuracy:", sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})


