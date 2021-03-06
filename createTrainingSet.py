#!/usr/bin/python
import h5py
import numpy as np
import re
from training_encodings import is_valid_encoding_type, valid_encoding_types
from itertools import compress

import os
import trainingRead as tr
import argparse

parser = argparse.ArgumentParser(description='Convert MinION fast5-reads into'
                                             'onehot vectors (in npy-files) with given properties,'
                                             'for training of neural networks to recognize'
                                             'those porperties.')
inputArg = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('-o', '--outFolder', type=str, required=True,
                    help='Folder in which results are stored')
parser.add_argument('-c', '--encoding-type', type=str, required=True,
                    help='Specify which kind of classification to adhere to. Must be one of the following types: %s'
                    % ', '.join(valid_encoding_types))
parser.add_argument('-n', '--normalization', type=str, required=False, default='median',
                    help='Specify how the raw data should be normalized.')
parser.add_argument('--use-nanoraw', action='store_true',
                    help='Use nanoraw-correction version of the sequence.')
parser.add_argument('--min-content-class', type=int, required=False, default=None,
                      help='Specify a class for which a lower bound in occurance should be defined.')
parser.add_argument('--min-content-percentage', type=float, required=False, default=None,
                      help='Specify a minimum percentage at which given class should occur. Discard reads that do not'
                           'fulfill the requirement.')
parser.add_argument('--deletion-affected-only', action='store_true',
                    help='Mark HP-events affected by deletions.')
inputArg.add_argument('-i', '--inputFolder', type=str, required=False,
                      help='Specify location of reads')
inputArg.add_argument('-l', '--inputList', type=str, required=False, nargs='*',
                      help='Specify list of reads')

args = parser.parse_args()

if not is_valid_encoding_type(args.encoding_type):
    raise ValueError('%s not recognized as a valid encoding type' % args.encoding_type)

outFolder = args.outFolder
if outFolder[-1] != '/':
    outFolder += '/'
if not os.path.isdir(outFolder):
    os.mkdir(outFolder)

if args.inputFolder is not None:
    if args.inputFolder[-1] != '/':
        args.inputFolder += '/'
    reads = os.listdir(args.inputFolder)
    reads = [args.inputFolder + r for r in reads]
else:
    reads = args.inputList

read_count = 0
success_count = 0
pattern = "(?<=read)\d+"
for read in reads:
    readnb = int(re.search(pattern, read).group())
    hdf = h5py.File(read, 'r')
    read_count += 1
    try:
        tr_cur = tr.TrainingRead(hdf, readnb, args.normalization, use_nanoraw=args.use_nanoraw)
    except KeyError:
        print('HDF5 key error encountered')
        hdf.close()
        continue

    if args.deletion_affected_only:
        encoded = tr_cur.classify_deletion_affected_events()
    else:
        encoded = tr_cur.classify_events(args.encoding_type)
    if tr_cur.events is not None and encoded is not None:
        unknown_index = [tc != 'NNNNN' for tc in tr_cur.events]  # Remove data for raw data without a k-mer
        onehot = list(compress(encoded, unknown_index))
        # onehot = encoded[unknown_index]
        frac_min_class = np.sum([oh == args.min_content_class for oh in onehot]) / len(encoded)
        if frac_min_class == 0. or args.min_content_percentage is not None and frac_min_class < args.min_content_percentage:
            print('Read discarded; positive example fraction too low.')
            continue

        outName = outFolder + os.path.basename(read)[:-6] + '_' + args.encoding_type + '.npz'
        np.savez(outName,
                 raw=list(compress(tr_cur.raw, unknown_index)),
                 # raw=tr_cur.raw[unknown_index],
                 onehot=onehot,
                 # base_labels=tr_cur.events[unknown_index]
                 base_labels=list(compress(tr_cur.events, unknown_index))
                 )
        success_count += 1
    hdf.close()
    if not read_count % 10:
        print("%d reads processed, %d training files created" % (read_count, success_count))
