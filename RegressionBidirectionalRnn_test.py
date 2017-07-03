from RegressionBidirectionalRnn import RegressionBidirectionalRnn
import os
import random

import reader
import helper_functions


from bokeh.io import save, output_file

# Define path of training dataset
tr_path = '/mnt/scratch/lanno001/readFiles/ecoliLoman/ecoliLoman_simts_hpp01_multiclass/'
tr_list = os.listdir(tr_path)

# Define path of test dataset
ts_path = '/mnt/scratch/lanno001/readFiles/ecoliLoman/ecoliLoman_simts_hpp01_multiclass/'
ts_list = os.listdir(ts_path)

# Define path to tensorboard parent directory
tb_path = '/mnt/scratch/lanno001/tensorboardFiles/regression/'

# Define path to additional graphs
graph_path = '/mnt/scratch/lanno001/hpTraceGraphs/regression/'

# Define hyper-parameters
batch_size = 64
read_length = 10000
input_size = 101  # make sure this is uneven so one position is in the middle!
num_layers = 2
cell_type='LSTM'
optimizer = 'adam'

layer_size = 101
num_classes = 1
num_epochs = 3

dropout_keep_prob = 1
training_iterations = 100000
learning_rate = 0.001
pos_weight = 1/1000

tst = RegressionBidirectionalRnn(batch_size, input_size, num_layers, read_length, cell_type, layer_size,
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
                ts_plot = helper_functions.plot_timeseries(raw, base_labels, y_hat, tst)
                output_file('%stimeseries_ordinal_ep%d_step%d.html' % (graph_path, epoch_index, tr_index))
                save(ts_plot)

