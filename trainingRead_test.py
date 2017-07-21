from trainingRead import TrainingRead
import re
import h5py
import os
import numpy as np

tr_path = '/mnt/scratch/lanno001/readFiles/ecoliLoman/ecoliLoman/'
tr_list = os.listdir(tr_path)

# tr_list = ['nanopore2_20170301_FNFAF09967_MN17024_sequencing_run_170301_MG1655_PC_RAD002_62645_ch137_read3942_strand.fast5']

for tr_name in tr_list:
    pattern = "(?<=read)\d+"
    readnb = int(re.search(pattern, tr_name).group())
    hdf = h5py.File(tr_path+tr_name, 'r')
    try:
        tr = TrainingRead(hdf, readnb, 'median', use_nanoraw=False)
    except KeyError:
        print('encountered key error')
        continue
    events = tr.events
    labels = tr.classify_events('hp_5class')
    labels = tr.classify_deletion_affected_events()
    if 5 in labels:
        print('read with deletion')

    else:
        print('read treated, no deletion')

