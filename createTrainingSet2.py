#!/usr/bin/python
import h5py
import numpy as np
import re
from itertools import compress
import os
import argparse

from training_encodings import valid_encoding_types
import trainingRead as tr
import readsim_model2 as rs

parser = argparse.ArgumentParser(description='Convert MinION fast5-reads into'
                                             'onehot vectors (in npy-files) denoting hompolymers affected by '
                                             'deletions.')
inputArg = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('-o', '--outFolder', type=str, required=True,
                    help='Folder in which results are stored')
parser.add_argument('-n', '--normalization', type=str, required=False, default='median',
                    help='Specify how the raw data should be normalized.')
parser.add_argument('-c', '--encoding-type', type=str, required=False, default='hp_5class',
                    help='Specify which kind of classification to adhere to. Must be one of the following types: %s'
                    % ', '.join(valid_encoding_types))
parser.add_argument('--deletion-affected', action='store_true',
                    help='Only mark events that were affected by a deletion.')
parser.add_argument('--oversample', type=float, required=False, default=None,
                    help='If specified, oversample target stretches until defined fraction is reached.')
inputArg.add_argument('-i', '--inputFolder', type=str, required=False,
                      help='Specify location of reads')
inputArg.add_argument('-l', '--inputList', type=str, required=False, nargs='*',
                      help='Specify list of reads')

args = parser.parse_args()

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

read_count = 0; success_count = 0; unsuitable_count = 0
pattern = "(?<=read)\d+"
for read in reads:
    try:
        readnb = int(re.search(pattern, read).group())
        hdf = h5py.File(read, 'r')
        read_count += 1
        tr_cur = tr.TrainingRead(hdf, readnb, args.normalization)
        if args.deletion_affected:
            encoded = tr_cur.classify_deletion_affected_events()
        else:
            encoded = tr_cur.classify_events(args.encoding_type)

        if tr_cur.events is None or encoded is None or 5 not in encoded:
            continue

        # Oversample target examples
        if args.oversample is not None:
            rs_cur = rs.readsim_model(tr_cur.condensed_events, 5)
            if not rs_cur.all_kmers_present:
                continue
            base_labels, raw, onehot = tr_cur.oversample(encoded, rs_cur, args.oversample)
        else:
            base_labels = tr_cur.events
            raw = tr_cur.raw
            onehot = tr_cur.expand_sequence(encoded)

        # Remove data for raw data without a k-mer
        comb_list = [(bl, ra, oh) for bl, ra, oh in zip(base_labels, raw, onehot) if bl != 'NNNNN']
        base_labels, raw, onehot = zip(*comb_list)
        # unknown_index = [tc != 'NNNNN' for tc in base_labels]
        # onehot = list(compress(encoded, unknown_index))
        # raw = list(compress(tr_cur.raw, unknown_index))
        # base_labels = list(compress(tr_cur.events, unknown_index))

        # Save training read
        outName = outFolder + os.path.basename(read)[:-6] + '_deletionAffected.npz'
        np.savez(outName,
                 raw=raw,
                 onehot=onehot,
                 base_labels=base_labels
                 )
        print('Valid training read created.')
        success_count += 1
        hdf.close()
        if not read_count % 10:
            print("%d reads processed, %d training files created %d reads unsuitable" % (read_count,
                                                                                         success_count,
                                                                                         unsuitable_count))
    except KeyError:
        print('Key error.')
        hdf.close()
        continue
    except IndexError:
        print('encountered index error, find cause later: read %s ' % read)
        hdf.close()
        continue
    except ValueError:
        print('First sample nanoraw occurs before first_sample basecaller. Skipping.')
        hdf.close()
        continue
