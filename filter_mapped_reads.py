import h5py
import os
import re
import shutil

mapped_reads_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_mappedReads.txt'
mapped_reads_folder = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_mapped/'
reads_folder = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman/'

if not os.path.isdir(mapped_reads_folder):
    os.mkdir(mapped_reads_folder)

reads_list = os.listdir(reads_folder)

with open(mapped_reads_path, 'r') as mr:
    mapped_reads = mr.readlines()

mapped_reads = [m[:-22] for m in mapped_reads]

read_nb_pattern = "(?<=read)\d+"
count = 0; success_count = 0
for read in reads_list:
    read_nb = int(re.search(read_nb_pattern, read).group())
    hdf = h5py.File(reads_folder+read, 'r')
    read_id = hdf['Raw/Reads/Read_%d' % read_nb].attrs['read_id'].astype(str)
    if read_id in mapped_reads:
        shutil.copy(reads_folder+read, mapped_reads_folder)
        success_count += 1
    hdf.close()
    count += 1
    if not count % 100:
        print('%d reads processed, %d reads selected' % (count, success_count))