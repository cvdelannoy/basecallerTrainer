import argparse
from bokeh.io import save, output_file
import h5py
from math import isnan
import numpy as np
import os
import random
import re
import yaml

from OrdinalBidirectionalRnn import OrdinalBidirectionalRnn
import reader
import helper_functions
import trainingRead



with open (parameter_file, 'r') as pf:
    params = yaml.load(pf)

# Define path of training dataset
tr_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_simtr2_p025_delAffected/'
tr_list = os.listdir(tr_path)

# Define path of test dataset
ts_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_tr_hp5class_clipped_deletionAffected/'
ts_list = os.listdir(ts_path)
random.shuffle(ts_list)
ts_list_idx = 0

# Define path of original fast5 dataset
original_path = '/mnt/scratch/lanno001/readFiles/ecoliLoman/ecoliLoman/'

# Define path to tensorboard parent directory
tb_path = '/mnt/nexenta/lanno001/nobackup/tensorboardFiles/ordinal_hp_real/discard_later/ecoliLoman_simtr2_p025'

# Define hyper-parameters
batch_size = 32
cell_type = 'LSTM'
learning_rate = 0.01
layer_size = 51
optimizer = 'adadelta'
num_layers = 3

read_length = 5000
input_size = 51  # make sure this is uneven so one position is in the middle!
num_classes = 5
num_epochs = 6
dropout_keep_prob = 0.5
training_iterations = 1000
adaptive_positive_weighting = True
simulate_on_the_fly = False
additional_plotting = True

min_content_percentage = 0.01
tr_list = tr_list[:training_iterations]


bases = ['A', 'C', 'G', 'T']
hps = [i*5 for i in bases]

# Define path to additional graphs
graph_path = ('/mnt/nexenta/lanno001/nobackup/hpTraceGraphs/real/ecoliLoman_simtr2_p025/ordinal_%s_batchSize'
              '%s_learningRate%s_layerSize%s_%s_numLayers%s_apw%s/' % (
                  cell_type,
                  batch_size,
                  learning_rate,
                  layer_size,
                  optimizer,
                  num_layers,
                  adaptive_positive_weighting))

graph_path_timeseries = graph_path+'timeseries/'
if not os.path.isdir(graph_path):
    os.makedirs(graph_path_timeseries)
roc_graph_name = graph_path + 'roc.html'
roc_list = []
metrics_file_name = graph_path + 'metrics.txt'
if os.path.isfile(metrics_file_name):
    os.remove(metrics_file_name)
metrics_file = open(metrics_file_name, "a+")
metrics_file.write("TPR\tTNR\tPPV\tqscore\tdels\tins\tmismatch\tmatch\n")

tst = OrdinalBidirectionalRnn(batch_size, input_size, num_layers, read_length, cell_type, layer_size,
                              optimizer, num_classes, learning_rate, dropout_keep_prob, adaptive_positive_weighting)
tst.initialize_model(params=None)

for epoch_index in range(1,num_epochs+1):
    random.shuffle(tr_list)
    file_writer = helper_functions.set_logfolder(tst, tb_path, epoch_index)
    tr_index = 0; ts_index = 0; low_hp_index = 0
    for tr in tr_list:
        if simulate_on_the_fly:
            readnb = int(re.search("(?<=read)\d+", tr).group())
            hdf = h5py.File(tr_path+tr, 'r')
            try:
                tr_cur = trainingRead.TrainingRead(hdf, readnb, 'median', use_nanoraw=False)
            except KeyError:
                hdf.close()
                continue
            encoded = tr_cur.classify_events('hp_5class')
            if tr_cur.events is None:
                continue
            unknown_index = tr_cur.events != 'NNNNN'  # Remove data for raw data without a k-mer
            raw = tr_cur.raw[unknown_index]
            onehot = encoded[unknown_index]
            frac_min_class = np.sum(onehot == 5) / encoded.size
            if frac_min_class < min_content_percentage or raw.size < read_length:
                continue
            print('read simulated')
        else:
            raw, onehot, _ = reader.npz_to_tf(tr_path+tr, read_length)
            if raw is None:
                continue
        try:
            tst.train_model(raw, onehot)
        except ValueError:
            print('Value error, skipping read')
            continue
        tr_index += 1
        if not tr_index % 10:
            onehot = [None]; raw = None
            while raw is None or np.sum(5 == onehot) < 20:
                ts_read = ts_list[ts_list_idx]
                ts_list_idx += 1
                if ts_list_idx == len(ts_list):
                    ts_list_idx = 0
                    random.shuffle(ts_list)
                raw, onehot, base_labels = reader.npz_to_tf(ts_path + ts_read, read_length)

            tb_summary, loss, y_hat, TPR, TNR, PPV = tst.evaluate_model(raw, onehot, base_labels, hps)
            y_hat = np.concatenate((np.repeat(np.NaN, tst.label_shift), y_hat.reshape(-1)))

            file_writer.add_summary(tb_summary, tr_index)
            nb_hp_raw = np.sum(onehot == 5)
            hp_ratio = nb_hp_raw / onehot.size
            print("Training step %d loss %f TPR %f TNR %f PPV %f HP points %d HP ratio %f " % (tr_index,
                                                                                               loss,
                                                                                               TPR,
                                                                                               TNR,
                                                                                               PPV,
                                                                                               nb_hp_raw,
                                                                                               hp_ratio))
            if not isnan(TPR) and not isnan(TNR):
                roc_list.append((TPR, TNR, epoch_index))
                roc_plot = helper_functions.plot_roc_curve(roc_list)
                output_file(roc_graph_name)
                save(roc_plot)
                read_properties = helper_functions.retrieve_read_properties(original_path, tr)
                metrics = [TPR, TNR, PPV] + read_properties
                metrics = '\t'.join([str(n) for n in metrics]) + '\n'
                metrics_file.write("%s" % metrics)

            if not tr_index % 100:
                # y_hat = tst.predict(raw)
                raw = raw[:len(y_hat)]
                base_labels = base_labels[:len(y_hat)]
                ts_plot = helper_functions.plot_timeseries(raw, base_labels, y_hat, tst)
                output_file('%stimeseries_ordinal_ep%d_step%d.html' % (graph_path,
                                                                       epoch_index,
                                                                       tr_index))
                save(ts_plot)
metrics_file.close()

# tf.reset_default_graph()
