from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import os
import sys
import math

import reader

tf.logging.set_verbosity(tf.logging.INFO)

# - LSTM-RNN
# - 3 layers
# - unidirectional
# - regularization using dropout

# Define path of training dataset
tr_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_simtr/'
tr_list = os.listdir(tr_path)

# Define path of test dataset
# te_path = ''
# te_list = os.listdir(te_path)

# Define hyper-parameters
num_epochs = 10
total_series_length = 10000
truncated_backprop_length = 100
state_size = 10
num_classes = 2
batch_size = 100
num_batches = total_series_length//batch_size//truncated_backprop_length
num_layers = 2
dropout_keep_prob = 0.5
learning_rate = 0.001
pos_weight = 100

# Create model

# Define and reshape placeholders
with tf.name_scope('input'):
    batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
    rawdata = tf.expand_dims(batchX_placeholder, [-1])
    batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
    labels = tf.reshape(batchY_placeholder, [-1])

with tf.name_scope('initial_state_definition'):
    init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])
    state_per_layer_list = tf.unstack(init_state, axis=0)
    rnn_tuple_state = tuple(
        [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
         for idx in range(num_layers)]
    )


with tf.name_scope('define_multilayer_cells'):
    cell_list = []
    for _ in range(num_layers):
        cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(state_size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
        cell_list.append(cell)
    cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell(cell_list, state_is_tuple=True)

with tf.name_scope('define_rnn'):
    states_series, current_state = tf.nn.dynamic_rnn(cell,
                                                     rawdata,
                                                     initial_state=rnn_tuple_state)
    states_series = tf.reshape(states_series, [-1, state_size])

with tf.name_scope('hidden_to_output'):
    W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)
    logits = tf.matmul(states_series, W2) + b2
    logits_series = tf.unstack(tf.reshape(logits, [batch_size, truncated_backprop_length, num_classes]), axis=1)
    predictions_series = [tf.nn.softmax(logit) for logit in logits_series]

# Evaluate performance: cross-entropy for training, classification accuracy for interpretation
with tf.name_scope('performance_evaluation'):
    losses = tf.nn.weighted_cross_entropy_with_logits(logits=logits[:,1], targets=tf.cast(labels,tf.float32), pos_weight=pos_weight)
    total_loss = tf.reduce_mean(losses)

    pos_labels = tf.reduce_sum(labels)

    correct_pred = tf.equal(tf.argmax(logits, axis=1), tf.cast(labels, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    correct_pred_pos = tf.logical_and(correct_pred, tf.equal(labels, tf.ones_like(labels, tf.int32)))
    TPR = (tf.reduce_sum(tf.cast(correct_pred_pos, tf.float32)) #//
           # tf.reduce_sum(tf.cast(tf.argmax(logits, axis=1), tf.float32))
           )

with tf.name_scope('training'):
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

with tf.Session() as sess:
    tf.summary.scalar("accuracy", accuracy)
    mergedStats = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter('/home/lanno001/PycharmProjects/basecallerTrainer/tfLogs/', sess.graph)
    sess.run(tf.global_variables_initializer())
    loss_list = []

    epoch_idx = 0
    while epoch_idx < num_epochs:
        tri = 0; ski = 0
        print("New data, epoch", epoch_idx)
        while tri < len(tr_list):
            _current_state = np.zeros((num_layers, 2, batch_size, state_size))
            x, y = reader.npz_to_tf(tr_path+tr_list[tri], total_series_length, batch_size)
            tri += 1
            if x is None:
                ski += 1
                if not ski % 100:
                    print("%d of %d reads skipped so far." %(ski, tri))
                continue
            for batch_idx in range(num_batches):
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length

                batchX = x[:, start_idx:end_idx]
                batchY = y[:, start_idx:end_idx]

                if batchY.sum() < batchY.size*0.01:
                    continue

                _total_loss, _train_step, _current_state, _predictions_series, _accuracy, _TPR, _pos_labels = sess.run(
                    [total_loss, train_step, current_state, predictions_series, accuracy, TPR, pos_labels],
                    feed_dict={
                        batchX_placeholder: batchX,
                        batchY_placeholder: batchY,
                        init_state: _current_state
                    })
                loss_list.append(_total_loss)

                # pos_labels_total += _pos_labels

                if not batch_idx % 100:
                    print("Step %d Batch loss %f Accuracy %f TPR %d pos %d" % (batch_idx, _total_loss, _accuracy, _TPR, _pos_labels))
        epoch_idx += 1