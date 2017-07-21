from trainingRead2 import TrainingRead
import re
import h5py
import os

tr_path = '/mnt/scratch/lanno001/readFiles/ecoliLoman/ecoliLoman/'
tr_list = os.listdir(tr_path)

tr_count = 0; hit_count = 0
for tr_name in tr_list:
    pattern = "(?<=read)\d+"
    readnb = int(re.search(pattern, tr_name).group())
    hdf = h5py.File(tr_path+tr_name, 'r')
    try:
        tr = TrainingRead(hdf, readnb, 'median')
    except KeyError:
        print('key error occurred')
        hdf.close()
        continue
    except ValueError:
        print('clipped events border error')
        hdf.close()
        continue
    events = tr.events
    labels = tr.classify_events()
    if 5 in labels:
        hit_count += 1
        print('hit!')
    tr_count += 1
    if not tr_count % 10:
        print('%d reads treated so far, %d hits' % (tr_count, hit_count))
    hdf.close()
