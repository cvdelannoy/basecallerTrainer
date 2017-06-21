from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import os

import reader
import helper_functions

tf.logging.set_verbosity(tf.logging.INFO)

# BLSTM approach according to
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py

# Path to tensorboard logs folder
tb_path = '/home/lanno001/PycharmProjects/basecallerTrainer/tfLogs/'

# Define path of training dataset
tr_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_simts_multiclass/'
tr_list = os.listdir(tr_path)

# Define path of test dataset
# te_path = ''
# te_list = os.listdir(te_path)

# Define hyper-parameters
batch_size = 64
read_length = 10000
input_size = 101  # make sure this is uneven so one position is in the middle!

layer_size_list = [2, 5, 50, 101]
num_classes = 2
num_epochs = 3

dropout_keep_prob = 0.5
training_iterations = 100000
learning_rate = 0.0001
pos_weight = 1/1000

# Derived hyperparameters
label_shift = (input_size - 1)//2
# Read length accounting for shifting input 1 position each step
read_length_extended = (read_length - input_size + 1) * input_size
nb_steps = read_length_extended // input_size // batch_size  # NOTE: floored division
# Alternatively:
# nb_steps = (read_length - input_size + 1) / batch_size

# Create model

for layer_size in layer_size_list:
    # Define and reshape placeholders
    with tf.name_scope('input'):
        x_placeholder = tf.placeholder(tf.float32, [batch_size, nb_steps, input_size])  # None for batch size
        x = tf.unstack(x_placeholder, nb_steps, 1)
        # x = tf.split(x_placeholder, num_or_size_splits=nb_steps, axis=1)
        y_placeholder = tf.placeholder(tf.int32, [batch_size, nb_steps])
        y = tf.unstack(y_placeholder, nb_steps, 1)

        # init_state = tf.placeholder(tf.float32, [batch_size, input_size])


    with tf.name_scope('construct_rnn'):

        # Define weights and biases
        w = {
            'to_single': tf.Variable(tf.random_normal([2*layer_size, num_classes]))
        }

        b = {
            'to_single': tf.Variable(np.zeros([num_classes], dtype=np.float32))
        }

        def blstm(x, w, b, layer_size):
            with tf.variable_scope('forward'):
                fwd_layer = tf.contrib.rnn.BasicLSTMCell(num_units=layer_size, forget_bias=1.0)
            with tf.variable_scope('backward'):
                bwd_layer = tf.contrib.rnn.BasicLSTMCell(num_units=layer_size, forget_bias=1.0)
            # blstm_out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=fwd_layer, cell_bw=bwd_layer,
            #                                                           inputs=x, dtype=tf.float32,
            #                                                           initial_state_fw=init_state,
            #                                                           initial_state_bw=init_state)
            blstm_out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=fwd_layer, cell_bw=bwd_layer,
                                                                      inputs=x, dtype=tf.float32)
            return [tf.matmul(bo, w['to_single']) + b['to_single'] for bo in blstm_out]

    with tf.name_scope('make_predictions'):
        predicted = blstm(x, w=w, b=b, layer_size=layer_size)

    # with tf.name_scope('hidden_to_output'):
    #     logits_series = tf.unstack(tf.reshape(logits, [batch_size, truncated_backprop_length, num_classes]), axis=1)
    #     predictions_series = [tf.nn.softmax(logit) for logit in logits_series]

    # Evaluate performance: cross-entropy for training, classification accuracy for interpretation
    with tf.name_scope('training'):
        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pr, labels=yi) for pr, yi in zip(predicted, y)]
        # losses = [tf.nn.weighted_cross_entropy_with_logits(
        #     logits=tf.nn.softmax(pr)[:,1],
        #     targets=tf.cast(yi, dtype=tf.float32),
        #     pos_weight=pos_weight) for pr, yi in zip(predicted, y)]
        mean_loss = tf.reduce_mean(tf.stack(losses))
        train = tf.train.AdagradOptimizer(learning_rate).minimize(mean_loss)

        # Summary performance metrics
        P = tf.cast(tf.reduce_sum(tf.stack(y)), dtype=tf.int64)
        N = tf.cast(nb_steps*batch_size - P, dtype=tf.int64)
        TP = tf.reduce_sum(tf.stack(
            [tf.count_nonzero(tf.argmax(pr, axis=1) *
                              tf.cast(la, dtype=tf.int64))
             for pr, la in zip(predicted, y)]
        ))
        TN = tf.reduce_sum(tf.stack(
            [tf.count_nonzero(
                (tf.argmax(pr, axis=1) - 1) * (tf.cast(la, dtype=tf.int64) - 1))
             for pr, la in zip(predicted, y)]
        ))
        TPR = TP / P
        TNR = TN / N
        # pos_labels = tf.reduce_sum(y)
        # correct_pred = tf.equal(tf.argmax(predicted, axis=1), tf.cast(y, tf.int64))
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # correct_pred_pos = tf.logical_and(correct_pred, tf.equal(y,
        #                                                          tf.ones_like(y, tf.int32)))
        # TPR = (tf.reduce_sum(tf.cast(correct_pred_pos, tf.float32))  # //
        #        # tf.reduce_sum(tf.cast(tf.argmax(logits, axis=1), tf.float32))
        #        )


    # Collect data for tensorboard
    with tf.name_scope('summaries'):
        tf.summary.scalar('cross_entropy', mean_loss)
        tf.summary.scalar('TPR', TPR)
        tf.summary.scalar('TNR', TNR)
        merged_stats = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0; read_index = 0; epoch_index = 1
        # Create new log folder for run
        file_writer = helper_functions.set_logfolder(sess, tb_path, batch_size, layer_size,
                                                     learning_rate, pos_weight, epoch_index)
        # Start training
        while epoch_index <= num_epochs:

            # Load data
            raw, onehot = reader.npz_to_tf(tr_path+tr_list[read_index], read_length)

            # reshape data into nb_steps x input_size matrix, 1 row per prediction
            # raw_reshape = np.zeros((batch_size, nb_steps, input_size))
            raw_reshape = np.zeros(read_length_extended)
            onehot_reshape = np.zeros(read_length - input_size + 1)
            # for i in range(batch_size*nb_steps):
            i = 0
            while i+input_size < raw.size:
                cur_row = raw[i:i+input_size]
                raw_reshape[i*input_size:i*input_size+input_size] = cur_row
                onehot_reshape[i] = onehot[i+label_shift]
                i += 1

            raw_reshape = raw_reshape[:nb_steps*input_size*batch_size]
            raw_reshape = raw_reshape.reshape(batch_size, nb_steps, input_size)
            onehot_reshape = onehot_reshape[:batch_size*nb_steps]
            onehot_reshape = onehot_reshape.reshape(batch_size, nb_steps)

            # Run optimization
            sess.run(train, feed_dict={
                x_placeholder: raw_reshape,
                y_placeholder: onehot_reshape
            })

            # evaluate every 10th step
            if step % 10 == 0:
                _merged_stats, _mean_loss, _TPR, _TNR = sess.run([merged_stats, mean_loss, TPR, TNR], feed_dict={
                                               x_placeholder: raw_reshape,
                                               y_placeholder: onehot_reshape
                })
                file_writer.add_summary(_merged_stats, step)

                print("Training step %d loss %f TPR %f TNR %f" % (step, _mean_loss, _TPR, _TNR))
            step += 1; read_index += 1
            if read_index >= len(tr_list):
                epoch_index += 1
                print("Start epoch %d" % epoch_index)
                file_writer = helper_functions.set_logfolder(sess, tb_path, batch_size, layer_size,
                                                             learning_rate, pos_weight, epoch_index)
                read_index = 0
                step = 0
