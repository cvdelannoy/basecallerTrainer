#!/usr/bin/python
import h5py
import numpy as np
import re
# import tinydb as tdb
import os
import trainingRead as tr
import argparse

parser = argparse.ArgumentParser(description='Convert Nanpore fast5-reads into'
                                             'onehot vectors (in npy-files) with given properties,'
                                             'for training of neural networks to recognize'
                                             'those porperties.')
parser.add_argument('-o', '--outFolder', type=str, required=True,
                    help='Folder in which results are stored')
parser.add_argument('-r', '--readFolder', type=str, required=True,
                    help='Specify location of reads')
parser.add_argument('s', '--referenceSam', type=str, required=False,
                    help='Sam-file produced by a mapper of choice using the provided reads')
args = parser.parse_args()

outFolder = args.outFolder
readdir = args.readFolder
refsam = args.refsam

# DEBUGGING
# dbname = "/media/carlos/Data/LocalData/dbs/trainingRead_ecoliSubset.json"
# outFolder = "/media/carlos/Data/LocalData/npzFiles"
# readdir = "/media/carlos/Data/LocalData/readFiles/ecoliLoman/ecoliSubset"
# refsam = "/media/carlos/Data/LocalData/refgenomes/ecoli_K12MG1655.fa"

# db = tdb.TinyDB(dbname)

pattern = "(?<=read)\d+"
for fname in os.listdir(readdir):
    read = readdir + "/" + fname
    readnb = int(re.search(pattern, read).group())
    hdf = h5py.File(read, 'r')
    tr_cur = tr.TrainingRead(hdf, readnb, refsam)

    np.savez(outFolder+'/'+fname[:-6]+'_onehot.npz',
             raw=tr_cur.rawsignal,
             onehot=tr_cur.homopolymer_onehot)

    # db.insert({'read': fname,
    #           'rawsignal': tr_cur.rawsignal.tolist(),
    #            'onehot': tr_cur.homopolymer_onehot.tolist()})
    hdf.close()
