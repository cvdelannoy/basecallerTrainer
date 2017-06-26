from ClassificationBidirectionalRnn import ClassificationBidirectionalRnn
import os

import reader

# Define path of training dataset
tr_path = '/mnt/scratch/lanno001/readFiles/ecoliLoman/ecoliLoman_simts_multiclass/'
tr_list = os.listdir(tr_path)


# Define hyper-parameters
batch_size = 64
read_length = 10000
input_size = 101  # make sure this is uneven so one position is in the middle!
num_layers = 2
cell_type='LSTM'
optimizer = 'adam'

layer_size = 101
num_classes = 4
num_epochs = 3

dropout_keep_prob = 1
training_iterations = 100000
learning_rate = 0.001
pos_weight = 1/1000

raw, onehot, base_labels = reader.npz_to_tf(tr_path+tr_list[0], read_length)

tst = ClassificationBidirectionalRnn(batch_size, input_size, num_layers, read_length, cell_type, layer_size,
                              optimizer, num_classes, learning_rate, dropout_keep_prob)

tst.initialize_model(None)
tst.train_model(raw, onehot)
