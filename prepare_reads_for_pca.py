import numpy as np
import os

"""
Prepare training reads for PCA analysis in R
"""


def reshape_raw(raw, input_size):
    """
    Reshape raw data vector into nb_steps x input_size numpy matrix
    """
    read_length_extended = (raw.size - input_size + 1) * input_size
    raw_reshape = np.zeros(read_length_extended)
    i = 0
    while i + input_size < raw.size:
        cur_row = raw[i:i + input_size]
        raw_reshape[i * input_size:i * input_size + input_size] = cur_row
        i += 1
    raw_reshape = raw_reshape.reshape(-1, input_size)
    return raw_reshape


def reshape_onehot(onehot, input_size):
    label_shift = (input_size - 1) // 2
    return onehot[label_shift:-label_shift]


read_length = 5000
input_size = 51
batch_size = 32

# tr_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_tr_hp5class/'
# tr_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_simtr_hp5class_hpProb01/'
tr_path= '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_tr_hp5class_noNanoraw_hp1_clipped/'
tr_list = os.listdir(tr_path)[:3]

out_path = '/home/lanno001/experimental/rnn_pca_analysis/data/allHPs/'

if not os.path.isdir(out_path):
    os.mkdir(out_path)

count = 0
for tr_name in tr_list:
    tr = np.load(tr_path+tr_name)
    raw = reshape_raw(tr['raw'], input_size)
    onehot = reshape_onehot(tr['onehot'], input_size)

    file_name = out_path + tr_name[:-4]
    np.savetxt(file_name + '_pca_labels.txt', onehot, fmt='%d')
    np.savetxt(file_name + '_pca_raw.txt', raw, fmt='%f')
    count += 1

    if not count % 10:
        print('%d reads created' % count)
