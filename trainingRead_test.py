from trainingRead import TrainingRead
import re
import h5py
import os

tr_path = '/mnt/scratch/lanno001/readFiles/ecoliLoman/ecoliLoman/'
tr_list = os.listdir(tr_path)

for tr_name in tr_list:
    pattern = "(?<=read)\d+"
    readnb = int(re.search(pattern, tr_name).group())
    hdf = h5py.File(tr_path+tr_name, 'r')
    try:
        tr = TrainingRead(hdf, readnb, 'median', use_nanoraw=False)
    except KeyError:
        continue
    events = tr.events
    labels = tr.classify_events('hp_5class')
