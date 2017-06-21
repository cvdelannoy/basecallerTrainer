#!/usr/bin/python
import h5py
import numpy as np
import re
# import tinydb as tdb
import os
import trainingRead as tr
import argparse

parser = argparse.ArgumentParser(description='Convert MinION fast5-reads into'
                                             'onehot vectors (in npy-files) with given properties,'
                                             'for training of neural networks to recognize'
                                             'those porperties.')
inputArg=parser.add_mutually_exclusive_group(required=True)
parser.add_argument('-o', '--outFolder', type=str, required=True,
                    help='Folder in which results are stored')
inputArg.add_argument('-i', '--inputFolder', type=str, required=False,
                    help='Specify location of reads')
inputArg.add_argument('-l', '--inputList', type=str, required=False, nargs='*',
                    help='Specify list of reads')
#parser.add_argument('-s', '--referenceSam', type=str, required=True,
#                    help='Sam-file produced by a mapper of choice using the provided reads')
parser.add_argument('--nanoraw-only', action='store_true',
                    help='Only return data if nanoraw was able to map and resquiggle the read.')
parser.add_argument('--with-pos-only', action='store_true',
                    help='Only return data if any matches with the supplied regex were found.')
args = parser.parse_args()

outFolder = args.outFolder
if outFolder[-1] != '/':
    outFolder += '/'
# refsam = args.referenceSam
if args.inputFolder is not None:
    if args.inputFolder[-1] != '/':
        args.inputFolder += '/'
    reads = os.listdir(args.inputFolder)
    reads = [args.inputFolder + r for r in reads]
else:
    reads = args.inputList


# DEBUGGING
# dbname = "/media/carlos/Data/LocalData/dbs/trainingRead_ecoliSubset.json"
# outFolder = "/media/carlos/Data/LocalData/npzFiles"
# readdir = "/media/carlos/Data/LocalData/readFiles/ecoliLoman/ecoliSubset"
# refsam = "/media/carlos/Data/LocalData/refgenomes/ecoli_K12MG1655.fa"
# db = tdb.TinyDB(dbname)

read_count = 0; success_count = 0
pattern = "(?<=read)\d+"
for read in reads:
    readnb = int(re.search(pattern, read).group())
    hdf = h5py.File(read, 'r')
    tr_cur = tr.TrainingRead(hdf, readnb, args.nanoraw_only, args.with_pos_only)

    if tr_cur.homopolymer_onehot is not None:
        outName = outFolder + os.path.basename(read)[:-6] + '_onehot.npz'
        np.savez(outName,
                 raw=tr_cur.rawsignal,
                 onehot=tr_cur.homopolymer_onehot)
        success_count += 1
    # db.insert({'read': fname,
    #           'rawsignal': tr_cur.rawsignal.tolist(),
    #            'onehot': tr_cur.homopolymer_onehot.tolist()})
    hdf.close()
    read_count += 1
    if not read_count % 100:
        print("%d reads processed, %d training files created" % (read_count, success_count))
