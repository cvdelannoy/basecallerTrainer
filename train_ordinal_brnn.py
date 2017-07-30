from bokeh.io import save, output_file
from math import isnan
import numpy as np
import os
import random
import yaml
import tensorflow as tf
import re

from OrdinalBidirectionalRnn import OrdinalBidirectionalRnn
import reader
import helper_functions

def main(args):
    # Load parameter file
    with open(args.rnn_parameter_file, 'r') as pf:
        params = yaml.load(pf)
    param_base_name = re.search('(?<=/)[^/]+(?=.yaml)', args.rnn_parameter_file).group()

    # Compose lists of training and test reads
    tr_list = helper_functions.parse_input_path(args.training_data)
    tr_list = tr_list[:params['training_iterations']]
    ts_list = helper_functions.parse_input_path(args.test_data)
    random.shuffle(ts_list)
    ts_list_idx = 0
    ts_endlist = random.sample(ts_list, 100)

    # Define path to tensorboard parent directory
    tb_parentdir = helper_functions.parse_output_path(args.tensorboard_path)

    # Define path to additional graphs
    graph_path = args.additional_graphs_path
    graph_path = helper_functions.parse_output_path(graph_path)
    graph_path = helper_functions.parse_output_path(graph_path + param_base_name)
    graph_path_timeseries = helper_functions.parse_output_path(graph_path + 'timeseries')
    roc_graph_name = graph_path + 'roc.html'
    roc_end_graph_name = graph_path + 'roc_end.html'
    roc_list = []
    roc_list_end = []

    # If original fast5 dataset is supplied, compose additional file with read quality metrics
    if args.original_data is not None:
        generate_quality_metrics_file = True
        metrics_file_name = graph_path + 'metrics.txt'
        if os.path.isfile(metrics_file_name):
            os.remove(metrics_file_name)
        with open(metrics_file_name, "a+") as metrics_file:
            metrics_file.write("TPR\tTNR\tPPV\tqscore\tdels\tins\tmismatch\tmatch\n")
    else:
        generate_quality_metrics_file = False

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

    # Create save-file for model if required
    if args.save_model:
        save_model_path = helper_functions.parse_output_path(args.save_model)
        saver = tf.train.Saver()
        checkpoint_name = save_model_path + param_base_name +'.ckpt'

    obrnn.initialize_model(params=args.model_weights)

    for epoch_index in range(1,params['num_epochs']+1):
        print('start epoch %d' % epoch_index)
        random.shuffle(tr_list)
        file_writer = helper_functions.set_logfolder(obrnn, param_base_name, tb_parentdir, epoch_index)
        tr_index = 0
        for tr in tr_list:
            raw, onehot, _ = reader.npz_to_tf(tr, params['read_length'])
            if raw is None:
                continue
            try:
                loss = obrnn.train_model(raw, onehot)
            except ValueError:
                print('Value error, skipping read')
                continue
            tr_index += 1
            if not tr_index % 10:
                onehot = [None]; raw = None; potential_targets = 0; real_targets = 0
                while raw is None or \
                                real_targets < 20 or \
                                        args.potential_targets is not None and potential_targets == real_targets:
                    ts_read = ts_list[ts_list_idx]
                    ts_list_idx += 1
                    if ts_list_idx == len(ts_list):
                        ts_list_idx = 0
                        random.shuffle(ts_list)
                    raw, onehot, base_labels = reader.npz_to_tf(ts_read, params['read_length'])
                    # Calculate how many targets are actually in the sequence
                    real_targets = np.sum(5 == onehot)
                    if args.potential_targets is None:
                        potential_targets = onehot.size
                    else:
                        potential_targets = sum([1 for bl in base_labels if bl in args.potential_targets])

                tb_summary, _, y_hat, TPR, TNR, PPV = obrnn.evaluate_model(raw, onehot,
                                                                              base_labels, args.potential_targets)

                y_hat = np.concatenate((np.repeat(np.NaN, obrnn.label_shift), y_hat))

                file_writer.add_summary(tb_summary, tr_index)
                nb_hp_raw = np.sum(onehot == 5)
                hp_ratio = nb_hp_raw / onehot.size
                print("Training step %d loss %f TPR %f TNR %f PPV %f HP points %d HP ratio %f targets %d" %
                      (tr_index, loss, TPR, TNR, PPV, nb_hp_raw, hp_ratio, potential_targets))
                if not isnan(TPR) and not isnan(TNR):
                    roc_list.append((TPR, TNR, epoch_index))
                    roc_plot = helper_functions.plot_roc_curve(roc_list)
                    output_file(roc_graph_name)
                    save(roc_plot)
                    if generate_quality_metrics_file:
                        read_properties = helper_functions.retrieve_read_properties(args.original_data, tr)
                        metrics = [TPR, TNR, PPV] + read_properties
                        metrics = '\t'.join([str(n) for n in metrics]) + '\n'
                        with open(metrics_file_name, "a+") as metrics_file:
                            metrics_file.write("%s" % metrics)
                if not tr_index % 100:
                    raw = raw[:len(y_hat)]
                    base_labels = base_labels[:len(y_hat)]
                    ts_plot = helper_functions.plot_timeseries(raw, base_labels, y_hat, obrnn)
                    output_file('%stimeseries_ordinal_ep%d_step%d.html' % (graph_path_timeseries,
                                                                           epoch_index,
                                                                           tr_index))
                    save(ts_plot)
        # At the end of each epoch, save model parameters
        if args.save_model:
            saver.save(obrnn.session, checkpoint_name, write_meta_graph=False)
        roc_list_cur = []
        for ts in ts_endlist:
            raw, onehot, base_labels = reader.npz_to_tf(ts, params['read_length'])
            real_targets = np.sum(5 == onehot)
            if args.potential_targets is None:
                potential_targets = onehot.size
            else:
                potential_targets = sum([1 for bl in base_labels if bl in args.potential_targets])

            if raw is None or real_targets < 20 or potential_targets == real_targets:
                continue
            _, _, _, TPR, TNR, PPV = obrnn.evaluate_model(raw, onehot, base_labels,
                                                                          args.potential_targets)
            roc_list_cur.append((TPR, TNR))
        TPR_cur = sum(r[0] for r in roc_list_cur) / len(roc_list_cur)
        TNR_cur = sum(r[1] for r in roc_list_cur) / len(roc_list_cur)
        roc_list_end.append((TPR_cur, TNR_cur, epoch_index))
    roc_plot = helper_functions.plot_roc_curve(roc_list_end)

    output_file(roc_end_graph_name)
    save(roc_plot)