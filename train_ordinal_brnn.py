from bokeh.io import save, output_file
from math import isnan
import numpy as np
import os
import random
import yaml

from OrdinalBidirectionalRnn import OrdinalBidirectionalRnn
import reader
import helper_functions

def main(args):
    bases = ['A', 'C', 'G', 'T']
    hps = [i * 5 for i in bases]

    with open(args.rnn_parameter_file, 'r') as pf:
        params = yaml.load(pf)

    # Compose lists of training and test reads
    tr_list = helper_functions.parse_input_path(args.training_data)
    tr_list = tr_list[:params['training_iterations']]
    ts_list = helper_functions.parse_input_path(args.test_data)
    random.shuffle(ts_list)
    ts_list_idx = 0

    # Define path to tensorboard parent directory
    tb_parentdir = helper_functions.parse_output_path(args.tensorboard_path)

    # Define path to additional graphs

    graph_path = args.additional_graphs_path

    graph_path_timeseries = graph_path+'timeseries/'
    if not os.path.isdir(graph_path):
        os.makedirs(graph_path_timeseries)
    roc_graph_name = graph_path + 'roc.html'
    roc_list = []

    # If original fast5 dataset is supplied, compose additional file with read quality metrics
    if args.original_data is not None:
        generate_quality_metrics_file = True
        metrics_file_name = graph_path + 'metrics.txt'
        if os.path.isfile(metrics_file_name):
            os.remove(metrics_file_name)
        metrics_file = open(metrics_file_name, "a+")
        metrics_file.write("TPR\tTNR\tPPV\tqscore\tdels\tins\tmismatch\tmatch\n")
    else:
        generate_quality_metrics_file = False

    # Construct and initialize rnn
    obrnn = OrdinalBidirectionalRnn(batch_size=params['batch_size'],
                                    input_size=params['input_size'],
                                    num_layers=params['num_layers'],
                                    read_length=params['read_length'],
                                    cell_type=params['cell_type'],
                                    layer_size=params['layer_size'],
                                    name_optimizer=params['optimizer'],
                                    num_classes=params['num_classes'],
                                    learning_rate=params['learning_rate'],
                                    dropout_keep_prob=params['dropout_keep_prob'],
                                    adaptive_positive_weighting=params['adaptive_positive_weighting'])
    obrnn.initialize_model(params=None)

    for epoch_index in range(1,params['num_epochs']+1):
        random.shuffle(tr_list)
        file_writer = helper_functions.set_logfolder(obrnn, tb_parentdir, epoch_index)
        tr_index = 0; ts_index = 0; low_hp_index = 0
        for tr in tr_list:
            raw, onehot, _ = reader.npz_to_tf(tr, params['read_length'])
            if raw is None:
                continue
            try:
                obrnn.train_model(raw, onehot)
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
                    raw, onehot, base_labels = reader.npz_to_tf(ts_read, params['read_length'])
                tb_summary, loss, y_hat, TPR, TNR, PPV = obrnn.evaluate_model(raw, onehot, base_labels, hps)
                y_hat = np.concatenate((np.repeat(np.NaN, obrnn.label_shift), y_hat.reshape(-1)))

                # base_labels_trunc = base_labels[obrnn.label_shift:]
                base_labels_trunc = base_labels[:y_hat.size]
                is_target_kmer = [1 if km in hps else 0 for km in base_labels_trunc]
                y_hat_pp = [1 if yh == 5 and itk == 0 else yh for yh,itk in zip(y_hat, is_target_kmer)]

                # onehot_pp = onehot[obrnn.label_shift:]
                onehot_pp = onehot[:y_hat.size]
                TP_pp = sum([1 if yhp == oh and yhp == 5 else 0 for yhp, oh in zip(y_hat_pp, onehot_pp)])
                TPR_pp = TP_pp / sum([ohp == 5 for ohp in onehot_pp])
                PPV_pp = TP_pp / sum([yhp == 5 for yhp in y_hat_pp])
                TN_pp = sum([1 if oh != 5 and yhp != 5 else 0 for yhp, oh in zip(y_hat_pp, onehot_pp)])
                TNR_pp = TN_pp / np.sum(onehot_pp != 5)

                file_writer.add_summary(tb_summary, tr_index)
                nb_hp_raw = np.sum(onehot == 5)
                hp_ratio = nb_hp_raw / onehot.size
                # print("Training step %d loss %f TPR %f TNR %f PPV %f HP points %d HP ratio %f " %
                #       (tr_index, loss, TPR, TNR, PPV, nb_hp_raw, hp_ratio))
                print("Training step %d loss %f TPR %f %f TNR %f %f PPV %f %f HP points %d HP ratio %f " %
                      (tr_index, loss, TPR, TPR_pp, TNR, TNR_pp, PPV, PPV_pp, nb_hp_raw, hp_ratio))
                if not isnan(TPR) and not isnan(TNR):
                    roc_list.append((TPR, TNR, epoch_index))
                    roc_plot = helper_functions.plot_roc_curve(roc_list)
                    output_file(roc_graph_name)
                    save(roc_plot)
                    if generate_quality_metrics_file:
                        read_properties = helper_functions.retrieve_read_properties(args.original_data, tr)
                        metrics = [TPR, TNR, PPV] + read_properties
                        metrics = '\t'.join([str(n) for n in metrics]) + '\n'
                        metrics_file.write("%s" % metrics)
                if not tr_index % 100:
                    raw = raw[:len(y_hat)]
                    base_labels = base_labels[:len(y_hat)]
                    ts_plot = helper_functions.plot_timeseries(raw, base_labels, y_hat, obrnn)
                    output_file('%stimeseries_ordinal_ep%d_step%d.html' % (graph_path,
                                                                           epoch_index,
                                                                           tr_index))
                    save(ts_plot)
    if generate_quality_metrics_file:
        metrics_file.close()
