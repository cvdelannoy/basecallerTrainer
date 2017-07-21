#!/usr/bin/python
from helper_functions import reader


# tr_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_simtr_hpp01_fiveclass_realNoise/'
# tr_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_tr_hp5class_noNanoraw_hp1_clipped/'
# tr_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_simtr_hpp01_equalEventLength_fiveclass_realNoise/'
# tr_path = '/mnt/scratch/lanno001/readFiles/ecoliLoman/ecoliLoman/'
tr_list = os.listdir(tr_path)

for tr in tr_list:
    raw, onehot, _ = reader.npz_to_tf(tr_path+tr, read_length)
    if raw is None:
        continue

    # Condense
    condensed_kmers = [[kmer, len(list(n)), 1] for kmer, n in itertools.groupby(self.events)]

    # Index marked events


    # Index non-marked, non 'NNNNN' events
