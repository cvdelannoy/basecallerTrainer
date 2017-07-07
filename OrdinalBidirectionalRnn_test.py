from OrdinalBidirectionalRnn import OrdinalBidirectionalRnn
import os
import random
import itertools
import tensorflow as tf
import numpy as np

import reader
import helper_functions

from bokeh.io import save, output_file

# Define path of training dataset
tr_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_tr_hp5class_noNanoraw/'
tr_list = os.listdir(tr_path)

# Define path of test dataset
ts_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_tr_hp5class_noNanoraw/'
ts_list = os.listdir(ts_path)

# Define path to tensorboard parent directory
tb_path = '/mnt/nexenta/lanno001/nobackup/tensorboardFiles/ordinal_hp_real/'

# Define hyper-parameters
batch_size = 32
cell_type='LSTM'
learning_rate = 0.1
layer_size = 51
optimizer = 'adadelta'
num_layers = 3

read_length = 5000
input_size = 51  # make sure this is uneven so one position is in the middle!
num_classes = 5
num_epochs = 4
dropout_keep_prob = 0.8
training_iterations = 100000
pos_weight = 1/1000

# # selected for parameter sweep
# batch_size_list = [64, 128, 32]
# cell_type_list = ['LSTM', 'GRU']
# learning_rate_list = [0.01, 0.1, 0.001]
# layer_size_list = [101, 51, 2]
# optimizer_list = ['adam', 'adadelta']
# num_layers_list = [2, 3]
#
# param_combos = itertools.product(batch_size_list,
#                                  cell_type_list,
#                                  learning_rate_list,
#                                  layer_size_list,
#                                  optimizer_list,
#                                  num_layers_list)
#
#
# restart_skip = 1
# restart_skip_counter = 0
#
# for (batch_size, cell_type, learning_rate, layer_size, optimizer, num_layers) in param_combos:
#     input_size = layer_size
#
#     restart_skip_counter += 1
#     if restart_skip_counter <= restart_skip:
#         continue

# Define path to additional graphs
graph_path = ('/mnt/nexenta/lanno001/nobackup/hpTraceGraphs/real/ordinal_%s_batchSize'
              '%s_learningRate%s_layerSize%s_%s_numLayers%s/' % (
                  cell_type,
                  batch_size,
                  learning_rate,
                  layer_size,
                  optimizer,
                  num_layers))
if not os.path.isdir(graph_path):
    os.mkdir(graph_path)

tst = OrdinalBidirectionalRnn(batch_size, input_size, num_layers, read_length, cell_type, layer_size,
                              optimizer, num_classes, learning_rate, dropout_keep_prob)
tst.initialize_model(params=None)

for epoch_index in range(num_epochs):
    random.shuffle(tr_list)
    file_writer = helper_functions.set_logfolder(tst, tb_path, epoch_index)
    tr_index = 0; ts_index = 0; low_hp_index = 0
    for tr in tr_list:
        raw, onehot, _ = reader.npz_to_tf(tr_path+tr, read_length)
        if raw is None:
            continue
        tst.train_model(raw, onehot)
        tr_index += 1
        if not tr_index % 10:
            ts_read = random.choice(ts_list)
            raw, onehot, base_labels = reader.npz_to_tf(ts_path + ts_read, read_length)
            if raw is None:
                continue
            fraction_hp = np.sum((onehot == 5)) / onehot.size
            if fraction_hp < 0.05:
                low_hp_index += 1
                if not low_hp_index % 10:
                    print('%d reads skipped due to low hp content so far' % low_hp_index)
                continue
            tb_summary, loss = tst.evalulate_model(raw, onehot)
            file_writer.add_summary(tb_summary, tr_index)
            print("Training step %d loss %f" % (tr_index, loss))
            if not tr_index % 100:
                y_hat = tst.predict(raw)
                raw = raw[:len(y_hat)]
                base_labels = base_labels[:len(y_hat)]
                ts_plot = helper_functions.plot_timeseries(raw, base_labels, y_hat, tst)
                output_file('%stimeseries_ordinal_ep%d_step%d.html' % (graph_path,
                                                                       epoch_index,
                                                                       tr_index))
                save(ts_plot)

# tf.reset_default_graph()