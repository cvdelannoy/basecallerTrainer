from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import os

# First attempt at hp detection:
# - bidirectional
# - dropout regularization

# Location training reads
training_path = '/media/carlos/Data/LocalData/npzFiles/'

# Define hyper-parameters
max_epochs = 10
total_series_length = 50000
truncated_backprop_length = 120
state_size = 120
num_classes = 2
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length
num_layers = 2
dropout_keep_prob = 0.5
output_size = 1  # number of predictions done per step
learning_rate = 0.001


# Arg check
if num_layers % 2:
    raise ValueError("Number of layers must be even")


def npz_to_tf(npz, total_series_length, name=None):
    """Turn npz-files into tf-readable objects

    :param npz: npz-file storing raw data and labels 
    :return: two tensors, containing raw data and labels
    """
    npz_read = np.load(npz)
    npz_read.close()

    # If read length is smaller than set series length, discard
    if npz_read['raw'].size < total_series_length:
        return None, None

    raw = npz_read['raw'][0:(total_series_length-1)]
    onehot = npz_read['onehot'][0:(total_series_length-1)]
    return raw, onehot


# Define placeholders
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

# Unstack initial state tensor to list of tensors for output at every step
state_per_layer_list = tf.unstack(init_state, 2, axis=0)
# state_per_layer_list = tf.unstack(init_state, axis=0)

# Then extend to allow storage for each layer
rnn_tuple_state = tuple(
    [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)]
)

#  Placeholders for logit weights and biases
W2 = tf.Variable(np.random.rand(output_size, truncated_backprop_length), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

# Construct network

# Define forward and backward cell
cell_fw = tf.contrib.rnn.BasicLSTMCell(num_layers/2, forget_bias=1.0)
cell_fw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=dropout_keep_prob)
cell_bw = tf.contrib.rnn.BasicLSTMCell(num_layers/2, forget_bias=1.0)
cell_bw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=dropout_keep_prob)

# construct blstm-rnn
outputs, current_state, _ = tf.contrib.rnn.static_bidirectional_rnn(
                                                        cell_fw, cell_bw,
                                                        state_per_layer_list,
                                                        dtype=tf.float32)

# calculate logits
logits = tf.matmul(W2, outputs[-1]) + b2
labels = tf.reshape(batchY_placeholder, [-1])

logits_series = tf.unstack(tf.reshape(logits,
                                      [batch_size, truncated_backprop_length, num_classes]), axis=1)
predictions_series = [tf.nn.softmax(logit) for logit in logits_series]

# Calculate cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
total_loss = tf.reduce_mean(cost)

# optimize
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# End network construction

# Initialize variables
init = tf.global_variables_initializer()

# List training reads
tr_list = os.listdir(training_path)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    read_count = 1
    loss_list = []
    while step * batch_size < max_epochs and read_count <= len(tr_list):
        while raw is None:
            raw, onehot = npz_to_tf(training_path+tr_list[read_count])
            read_count += 1
            if read_count > len(tr_list):
                raise ValueError("Ran out of reads")

        _current_state = np.zeros((num_layers, 2, batch_size, state_size))

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = raw[:, start_idx:end_idx]
            batchY = onehot[:, start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    init_state: _current_state
                })
            loss_list.append(_total_loss)
            if batch_idx % 100 == 0:
                print("Step", batch_idx, "Batch loss", _total_loss)
        step += 1
        raw = None
    print("Optimization Finished!")

