import itertools
import yaml
import argparse
import os

from helper_functions import parse_output_path

parser = argparse.ArgumentParser(description='Generate one or more yaml parameter files, required for to run'
                                             'RNNs in this library. One file is generated for every combination'
                                             'of arguments.')
parser.add_argument('--save-path', type=str,required=True,
                    help='Folder where parameter files are saved. Is made if not existing.')
parser.add_argument('--batch-size-list', type=int, required=True, nargs='+')
parser.add_argument('--cell-type-list', type=str, required=True, nargs='+')
parser.add_argument('--learning-rate-list', type=float, required=True, nargs='+')
parser.add_argument('--layer-size-list', type=int, required=True, nargs='+')
parser.add_argument('--optimizer-list', type=str, required=True, nargs='+')
parser.add_argument('--num-layers-list', type=int, required=True, nargs='+')
parser.add_argument('--read-length-list', type=int, required=True, nargs='+')
parser.add_argument('--input-size-list', type=int, required=False, default=[None], nargs='+')
parser.add_argument('--num-classes-list', type=int, required=True, nargs='+')
parser.add_argument('--num-epochs-list', type=int, required=True, nargs='+')
parser.add_argument('--dropout-keep-prob-list', type=float, required=True, nargs='+')
parser.add_argument('--training-iterations-list', type=int, required=True, nargs='+')
parser.add_argument('--additional-plotting-list', type=bool, required=True, nargs='+')
parser.add_argument('--adaptive-positive-weighting-list', type=bool, required=True, nargs='+')
parser.add_argument('--min-content-percentage-list', type=float, required=True, nargs='+')

args = parser.parse_args()

save_path = parse_output_path(args.save_path)


param_combos = itertools.product(args.batch_size_list,
                                 args.cell_type_list,
                                 args.learning_rate_list,
                                 args.layer_size_list,
                                 args.optimizer_list,
                                 args.num_layers_list,
                                 args.read_length_list,
                                 args.input_size_list,
                                 args.num_classes_list,
                                 args.num_epochs_list,
                                 args.dropout_keep_prob_list,
                                 args.training_iterations_list,
                                 args.additional_plotting_list,
                                 args.adaptive_positive_weighting_list,
                                 args.min_content_percentage_list
                                 )

for (batch_size,
     cell_type,
     learning_rate,
     layer_size,
     optimizer,
     num_layers,
     read_length,
     input_size,
     num_classes,
     num_epochs,
     dropout_keep_prob,
     training_iterations,
     adaptive_positive_weighting,
     additional_plotting,
     min_content_percentage) in param_combos:

    if input_size is None:
        input_size = layer_size
    param_dict = {
        'batch_size': batch_size,
        'cell_type': cell_type,
        'learning_rate': learning_rate,
        'layer_size': layer_size,
        'optimizer': optimizer,
        'num_layers': num_layers,
        'read_length': read_length,
        'input_size': input_size,
        'num_classes': num_classes,
        'num_epochs': num_epochs,
        'dropout_keep_prob': dropout_keep_prob,
        'training_iterations': training_iterations,
        'adaptive_positive_weighting': adaptive_positive_weighting,
        'additional_plotting': additional_plotting,
        'min_content_percentage': min_content_percentage
    }

    yaml_name = '%s_%s_bs%d_lr%.4f_ls%d_numl%d_rl%d_is%d_drop%.3f_apw%s.yaml' % (cell_type, optimizer,
                                                                             batch_size, learning_rate,
                                                                             layer_size, num_layers,
                                                                             read_length, input_size,
                                                                             dropout_keep_prob,
                                                                             adaptive_positive_weighting)
    yaml_name = save_path + yaml_name
    with open(yaml_name, 'w') as yaml_file:
        yaml.dump(param_dict, yaml_file,default_flow_style=False)


