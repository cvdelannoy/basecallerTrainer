import yaml
import h5py

import helper_functions
from reader import fast5_to_tf
from OrdinalBidirectionalRnn import OrdinalBidirectionalRnn

def main(args):
    with open(args.rnn_parameter_file, 'r') as pf:
        params = yaml.load(pf)

    # Compose lists of to-be corrected reads
    read_list = helper_functions.parse_input_path(args.training_data)


    # Set up model
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

    # Initialize with given model weights
    obrnn.initialize_model(args.model_weights)
    for read in read_list:
        # Retrieve raw data points
        raw, base_labels = fast5_to_tf(read)

        # Make predictions on deletion occurance in hompolymers


        # Merge base_labels into a single sequence


        # Add fastq header


        # Save as fastq

