#!/usr/bin/python
import argparse
import os
import numpy as np

from training_encodings import valid_encoding_types, class_number
import reader

parser = argparse.ArgumentParser(description='Change training read encoding of existing training reads.')
inputArg = parser.add_mutually_exclusive_group(required=True)
inputArg.add_argument('-i', '--inputFolder', type=str, required=False,
                      help='Specify location of reads')
inputArg.add_argument('-l', '--inputList', type=str, required=False, nargs='*',
                      help='Specify list of reads')
parser.add_argument('-o', '--outFolder', type=str, required=True,
                    help='Folder in which results are stored')
parser.add_argument('-c', '--encoding-type', type=str, required=False, default='hp_5class',
                    help='Specify which kind of classification to adhere to. Must be one of the following types: %s'
                    % ', '.join(valid_encoding_types))
# parser.add_argument('--deletion-affected', action='store_true',
#                     help='Only mark events that were affected by a deletion.')

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

read_count = 0
for read in reads:
    raw, _, base_labels = reader.npz_to_tf(read)
    onehot = [class_number(bl, args.encoding_type) for bl in base_labels]
    outName = outFolder + os.path.basename(read)
    np.savez(outName,
             raw=raw,
             onehot=onehot,
             base_labels=base_labels
             )
    read_count += 1
    if not read_count % 10:
        print('%d reads converted' % read_count)
