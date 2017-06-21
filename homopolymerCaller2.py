import tensorflow as tf
import reader

batch_size = 200
learning_rate = 0.001
training_epochs = 10

read = 'test.npz'


raw, labels = reader.npz_to_tf(read,batch_size)

# raw = tf.placeholder(tf.float16, shape = (batch_size,1))
# labels = tf.placeholder(tf.float32, shape = (1,1))

# Return fully connected layer
def fulconn_layer(input_data, output_dim, activation_func=None):
    input_dim = int(input_data.get_shape()[1])
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([output_dim]))
    if activation_func:
        return activation_func(tf.matmul(input_data, W) + b)
    else:
        return tf.matmul(input_data, W) + b

# define forward and backward cell
lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(1, forget_bias=1.0, state_is_tuple=True)
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(1, forget_bias=1.0, state_is_tuple=True)

# construct bidirectoinal
outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                  lstm_bw_cell,
                                                  inputs=raw,
                                                  time_major=False,
                                                  dtype=tf.float32)
rnn_layer1 = tf.TensorArray.unstack(tf.transpose(outputs, [0, 1]))[-1]
labels_hat = fulconn_layer(rnn_layer1, batch_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels_hat, labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(labels_hat, 1)), tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for epoch in range(training_epochs):
    # for i in range(int(mnist.train.num_examples/batch_size)):
    #     x_batch, y_batch = mnist.train.next_batch(batch_size)
    #     x_batch = x_batch.reshape([batch_size, s, n])
    #     sess.run(optimizer, feed_dict={x: x_batch, y: y_batch})
    sess.run(optimizer)
    train_accuracy = sess.run(accuracy)
    # train_accuracy = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch})
    # x_test = mnist.test.images.reshape([-1, s, n])
    # y_test = mnist.test.labels
    # test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
    print("epoch: %d, train_accuracy: %3f" % (epoch, train_accuracy))