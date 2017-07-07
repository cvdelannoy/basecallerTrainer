# from OrdinalBidirectionalRnn import OrdinalBidirectionalRnn
from ClassificationBidirectionalRnn import ClassificationBidirectionalRnn
import os
import random
import itertools
import tensorflow as tf

import reader
import helper_functions

from bokeh.io import save, output_file

# Define path of training dataset
tr_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_simtr_hpp01_basecallTrain1_noise1/'
tr_list = os.listdir(tr_path)

# Define path of test dataset
ts_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_simtr_hpp01_basecallTrain1_noise1/'
ts_list = os.listdir(ts_path)

# Define path to tensorboard parent directory
tb_path = '/mnt/nexenta/lanno001/nobackup/tensorboardFiles/classification_basecallTrain_4class/'
if not os.path.isdir(tb_path):
    os.mkdir(tb_path)

# Define hyper-parameters
batch_size = 128
cell_type='LSTM'
learning_rate = 0.01
layer_size = 101
optimizer = 'adam'
num_layers = 3

read_length = 10000
input_size = 101  # make sure this is uneven so one position is in the middle!
num_classes = 5
num_epochs = 2
dropout_keep_prob = 0.8
training_iterations = 100000
pos_weight = 1/1000


# Define path to additional graphs
graph_path = ('/mnt/nexenta/lanno001/nobackup/hpTraceGraphs/ClassificationBasecallTrainPuPy_%s_batchSize'
              '%s_learningRate%s_layerSize%s_%s_numLayers%s/' % (
                  cell_type,
                  batch_size,
                  learning_rate,
                  layer_size,
                  optimizer,
                  num_layers))
if not os.path.isdir(graph_path):
    os.mkdir(graph_path)

tst = ClassificationBidirectionalRnn(batch_size, input_size, num_layers, read_length, cell_type, layer_size,
                              optimizer, num_classes, learning_rate, dropout_keep_prob)
tst.initialize_model(params=None)

for epoch_index in range(num_epochs):
    random.shuffle(tr_list)
    file_writer = helper_functions.set_logfolder(tst, tb_path, epoch_index)
    tr_index = 0; ts_index = 0
    for tr in tr_list:
        raw, onehot, _ = reader.npz_to_tf(tr_path+tr, read_length)
        tst.train_model(raw, onehot)
        tr_index += 1
        if not tr_index % 10:
            ts_read = random.choice(ts_list)
            raw, onehot, base_labels = reader.npz_to_tf(ts_path + ts_read, read_length)
            tb_summary, loss = tst.evalulate_model(raw, onehot)
            file_writer.add_summary(tb_summary, tr_index)
            print("Training step %d loss %f" % (tr_index, loss))
            if not tr_index % 100:
                ts_read = random.choice(ts_list)
                raw, onehot, base_labels = reader.npz_to_tf(ts_path + ts_read, read_length)
                y_hat = tst.predict(raw)
                raw = raw[:len(y_hat)]
                base_labels = base_labels[:len(y_hat)]
                # base_labels = [x[len(x)//2] for x in base_labels]
                base_labels = onehot[:len(y_hat)]
                ts_plot = helper_functions.plot_timeseries(raw, base_labels, y_hat, tst, True)
                output_file('%stimeseries_categorical_ep%d_step%d.html' % (graph_path,
                                                                       epoch_index,
                                                                       tr_index))
                save(ts_plot)
# tf.reset_default_graph()