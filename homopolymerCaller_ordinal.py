from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import os
from math import pi

from bokeh.models import ColumnDataSource, CategoricalColorMapper, LabelSet, Range1d
from bokeh.plotting import figure, save, output_file
from bokeh.io import show

import cas.RWACell as rwa

import reader
import helper_functions



# import argparse
# parser = argparse.ArgumentParser(description='train an rnn to detect homopolymer regions')
# parser.add_argument('--learning-rate', type=float, required=True)
# parser.add_argument('--layer-size', type=int ,required=True)


tf.logging.set_verbosity(tf.logging.INFO)

# Adapted from homopolymerCaller_multiclass.py
# Adaptations marked for easier conversion to OO later on.

# Path to tensorboard logs folder
tb_path = '/home/lanno001/PycharmProjects/basecallerTrainer/tfLogs/'
# Subfolder for additional graphs
graph_path = tb_path + 'additional_graphs/'

# Define path of training dataset
tr_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_simts_multiclass_HpProb01/'
tr_list = os.listdir(tr_path)

# Define path of test dataset
# te_path = ''
# te_list = os.listdir(te_path)

# Define hyper-parameters
batch_size = 64
read_length = 10000
input_size = 101  # make sure this is uneven so one position is in the middle!
num_layers = 2
cell_type='LSTM'

layer_size = 101
num_classes = 4
num_epochs = 3

dropout_keep_prob = 1
training_iterations = 100000
learning_rate = 0.001
pos_weight = 1/1000

# Derived parameters
label_shift = (input_size - 1)//2
# Read length accounting for shifting input 1 position each step
read_length_extended = (read_length - input_size + 1) * input_size
nb_steps = read_length_extended // input_size // batch_size  # NOTE: floored division

# Create model
with tf.name_scope('input'):
    x_placeholder = tf.placeholder(tf.float32, [batch_size, nb_steps, input_size])
    x = tf.unstack(x_placeholder, nb_steps, 1)
    y_placeholder = tf.placeholder(tf.int32, [batch_size, nb_steps, num_classes])
    y = tf.unstack(y_placeholder, nb_steps, 1)

with tf.name_scope('construct_rnn'):
    # Define weights and biases
    w = {
        'to_single': tf.Variable(tf.random_normal([2*layer_size, num_classes]))
    }
    b = {
        'to_single': tf.Variable(np.zeros([num_classes], dtype=np.float32))
    }

    def blstm(x, w, b, layer_size, cell_type):
        def base_cell(cell_type, layer_size):
            if cell_type == 'GRU':
                return tf.contrib.rnn.GRUCell(num_units=layer_size)
            if cell_type == 'LSTM':
                return tf.contrib.rnn.BasicLSTMCell(num_units=layer_size, forget_bias=1.0)
            if cell_type == 'RWA':
                return rwa.RWACell(num_units=layer_size, decay_rate=0.0)
            return ValueError('Invalid cell_type given.')
        fwd_cell_list = []; bwd_cell_list=[]
        for fl in range(num_layers):
            with tf.variable_scope('forward%d' % fl, reuse=True):
                fwd_cell = base_cell(cell_type=cell_type, layer_size=layer_size)
                fwd_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(fwd_cell, output_keep_prob=dropout_keep_prob)
            fwd_cell_list.append(fwd_cell)
        fwd_multicell = tf.contrib.rnn.MultiRNNCell(fwd_cell_list)
        for bl in range(num_layers):
            with tf.variable_scope('backward%d' % bl, reuse=True):
                bwd_cell = base_cell(cell_type=cell_type, layer_size=layer_size)
                bwd_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(bwd_cell, output_keep_prob=dropout_keep_prob)
            bwd_cell_list.append(bwd_cell)
        bwd_multicell = tf.contrib.rnn.MultiRNNCell(bwd_cell_list)
        blstm_out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=fwd_multicell, cell_bw=bwd_multicell,
                                                                  inputs=x, dtype=tf.float32)
        return [tf.matmul(bo, w['to_single']) + b['to_single'] for bo in blstm_out]

with tf.name_scope('make_predictions'):
    y_hat = blstm(x, w=w, b=b, layer_size=layer_size, cell_type=cell_type)


with tf.name_scope('training'):
    # CHANGE: from softmax to sigmoid, cast labels to float
    losses = [tf.nn.sigmoid_cross_entropy_with_logits(logits=pr, labels=tf.cast(yi, dtype=tf.float32)) for pr, yi in zip(y_hat, y)]
    # losses = [tf.nn.weighted_cross_entropy_with_logits(
    #     logits=tf.nn.softmax(pr)[:,1],
    #     targets=tf.cast(yi, dtype=tf.float32),
    #     pos_weight=pos_weight) for pr, yi in zip(predicted, y)]
    mean_loss = tf.reduce_mean(tf.stack(losses))
    train = tf.train.AdamOptimizer(learning_rate).minimize(mean_loss)

with tf.name_scope('Performance_assessment'):
    # Summary performance metrics
    y_stacked = tf.reduce_sum(tf.stack(y, axis=-1), axis=1)
    y_hat_stacked = tf.reduce_sum(tf.stack(y_hat, axis=-1), axis=1)
    y_hat_stacked = tf.cast(tf.round(y_hat_stacked), dtype=tf.int32)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_stacked, y_hat_stacked), dtype=tf.float32))
    y_hat_out = tf.reshape(y_hat_stacked, [1,-1])

    # per class accuracy
    accuracy_list = []
    for i in range(num_classes):
        cm = tf.constant(i, shape=[batch_size, nb_steps], dtype=tf.int32)
        y_bin = tf.cast(tf.equal(y_stacked, cm), dtype=tf.int32)
        y_hat_bin = tf.cast(tf.equal(y_hat_stacked, cm), dtype=tf.int32)
        accuracy_class =  tf.reduce_mean(tf.cast(tf.equal(y_bin, y_hat_bin), dtype=tf.float32))
        accuracy_list.append(accuracy_class)
        # TPR and TNR for HPs
        if i == (num_classes - 1):
            TP = tf.count_nonzero(tf.multiply(y_bin, y_hat_bin))
            TN = tf.count_nonzero(tf.multiply(y_bin - 1, y_hat_bin - 1))
            TPR = TP / tf.count_nonzero(y_bin)
            TNR = TN / tf.count_nonzero(y_bin - 1)

# Collect data for tensorboard
with tf.name_scope('tensorboard_summaries'):
    tf.summary.scalar('cross_entropy', mean_loss)
    tf.summary.scalar('TPR', TPR)
    tf.summary.scalar('TNR', TNR)
    tf.summary.scalar('accuracy', accuracy)
    for i in range(num_classes):
        tf.summary.scalar('accuracy_class%d' % i, accuracy_list[i])
    merged_stats = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    step = 0; read_index = 0; epoch_index = 1
    # Create new log folder for run
    file_writer = helper_functions.set_logfolder(sess, tb_path, batch_size, layer_size,
                                                 learning_rate, pos_weight, epoch_index, cell_type)
    # Start training
    while epoch_index <= num_epochs:

        # Load data
        raw, onehot, base_labels = reader.npz_to_tf(tr_path+tr_list[read_index], read_length)

        # reshape data into nb_steps x input_size matrix, 1 row per prediction
        raw_reshape = np.zeros(read_length_extended)
        onehot_reshape = np.zeros((read_length - input_size + 1, num_classes),dtype=int)
        # onehot_reshape = np.zeros((read_length - input_size + 1),dtype=int)
        i = 0
        while i+input_size < raw.size:
            cur_row = raw[i:i+input_size]
            raw_reshape[i*input_size:i*input_size+input_size] = cur_row
            cur_onehot = np.zeros(num_classes, dtype=int)
            cur_onehot[0:onehot[i+label_shift]] = 1
            onehot_reshape[i,:] = cur_onehot
            # onehot_reshape[i] = onehot[i+label_shift]
            i += 1

        raw_reshape = raw_reshape[:nb_steps*input_size*batch_size]
        raw_reshape = raw_reshape.reshape(batch_size, nb_steps, input_size)
        onehot_reshape = onehot_reshape[:batch_size*nb_steps,:]
        onehot_reshape = onehot_reshape.reshape(batch_size, nb_steps, num_classes)


        # Run optimization
        sess.run(train, feed_dict={
            x_placeholder: raw_reshape,
            y_placeholder: onehot_reshape
        })

        # evaluate every 10th step
        if step % 10 == 0:
            _merged_stats, _mean_loss, _accuracy, _TPR, _TNR, y_hat = sess.run([merged_stats,
                                                                                mean_loss,
                                                                                accuracy,
                                                                                TPR, TNR,
                                                                                y_hat_out],
                                                                               feed_dict={
                x_placeholder: raw_reshape,
                y_placeholder: onehot_reshape
            })
            file_writer.add_summary(_merged_stats, step)
            print("Training step %d loss %f accuracy %f TPR %f TNR %f" % (step, _mean_loss, _accuracy, _TPR, _TNR))
            if step % 100 == 0:
                # Plot timeseries overlaid with classification
                ts_plot = figure(title='Classified time series')
                ts_plot.grid.grid_line_alpha = 0.3
                ts_plot.xaxis.axis_label = 'nb events'
                ts_plot.yaxis.axis_label = 'current signal'
                y_range = raw.max() - raw.min()
                colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
                col_mapper = CategoricalColorMapper(factors=list(range(num_classes+1)), palette=colors)
                source = ColumnDataSource(dict(
                    raw=raw[(label_shift - 1):][:y_hat.size],
                    event=list(range(batch_size * nb_steps)),
                    cat=y_hat[0, :],
                    cat_height=np.repeat(np.mean(raw[(label_shift - 1):][:y_hat.size]), batch_size * nb_steps),
                    base_labels= base_labels[(label_shift - 1):][:y_hat.size]
                ))
                ts_plot.rect(x='event', y='cat_height', width=1, height=y_range, source=source,
                             fill_color={
                                 'field': 'cat',
                                 'transform': col_mapper
                             },
                             line_color=None)
                base_labels_LabelSet = LabelSet(x='event', y='cat_height',
                                                y_offset=-y_range, angle=-0.5*pi,
                                                text='base_labels', text_baseline='middle',
                                                source=source)
                ts_plot.add_layout(base_labels_LabelSet)
                ts_plot.scatter(x='event', y='raw', source=source)
                ts_plot.plot_width = 10000
                ts_plot.plot_height = 500
                ts_plot.x_range = Range1d(0,500)
                output_file('%stimeseries_ep%d_step%d_TPR%f.html' % (graph_path, epoch_index, step, _TPR))
                save(ts_plot)
        step += 1; read_index += 1
        if read_index >= len(tr_list):
            epoch_index += 1
            print("Start epoch %d" % epoch_index)
            file_writer = helper_functions.set_logfolder(sess, tb_path, batch_size, layer_size,
                                                         learning_rate, pos_weight, epoch_index, cell_type)
            read_index = 0
            step = 0
