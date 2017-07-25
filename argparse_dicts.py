import argparse
import os

# = ( , {
#     'type': ,
#     'required': ,
#     'help':
# })

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

rnn_training_data = ('--training-data', {
    'type': str,
    'required': True,
    'nargs': '+',
    'help': 'npz-training reads on which the rnn should be trained. '
            'May be {a folder with/an array of/a single} training file.'
})

rnn_test_data = ('--test-data', {
    'type': str,
    'required': True,
    'help': 'npz-training reads on which rnn performance is evaluated. '
            'May be {a folder with/an array of/a single} training file.'
})

rnn_original_data = ('--original-data' , {
    'type': str,
    'required': False,
    'default': None,
    'help': 'Original fast5-files on which test data is based. If supplied,'
            'rnn performance measures and read quality measures are collected'
            'and stored.'
})

tensorboard_path = ('--tensorboard-path', {
    'type': str,
    'required': False,
    'default':'~/tensorboard_logs/',
    'help': 'Define different location to store tensorboard files. Default is home/tensorboard_logs/'
            'Folders and sub-folders are generated if not existiing.'
})

additional_graphs_path = ('--additional-graphs-path' , {
    'type': str,
    'required': False,
    'default': '~/rnn_additional_graphs/',
    'help': 'Define different location to store additional graphs, if made. Default is home/rnn_aditional_graphs/'
            'Folders and sub-folders are generated if not existiing.'
})

rnn_parameter_file = ('--rnn-parameter-file', {
    'type': str,
    'required': False,
    'default': os.path.join(__location__,'RnnParameterFile_defaults.yaml'),
    'help': 'a yaml-file containing parameters. If none supplied, default values are used.'})

def get_brnn_parser():
    parser = argparse.ArgumentParser(description='Train a bidirectional rnn to recognize patterns in MinION data '
                                                 'as stored by the labeling in a given training data set.')
    parser.add_argument(rnn_training_data[0], **rnn_training_data[1])
    parser.add_argument(rnn_test_data[0], **rnn_test_data[1])
    parser.add_argument(rnn_original_data[0], **rnn_original_data[1])
    parser.add_argument(tensorboard_path[0], **tensorboard_path[1])
    parser.add_argument(additional_graphs_path[0], **additional_graphs_path[1])
    parser.add_argument(rnn_parameter_file[0], **rnn_parameter_file[1])

    return parser

if __name__ == '__main__':
    raise ValueError('argument parser file, do not call.')