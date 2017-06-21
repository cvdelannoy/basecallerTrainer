#!/usr/bin/python
import argparse
import os
parser = argparse.ArgumentParser(description='Convert MinION fast5-reads into'
                                             'onehot vectors (in npy-files) with given properties,'
                                             'for training of neural networks to recognize'
                                             'those porperties.')
inputArg=parser.add_mutually_exclusive_group(required=True)
inputArg.add_argument('-i', '--inputFolder', type=str, required=False,
                    help='Specify location of reads')
inputArg.add_argument('-l', '--inputList', type=str, required=False,
                      nargs='*',
                      help='Specify list of reads')

args = parser.parse_args()

if args.inputList is not None:
    for n in args.inputList:
        print(n)

if args.inputFolder is not None:
    print(os.listdir(args.inputFolder))

if args.inputFolder is None and args.inputList is None:
    raise ValueError('test')

print('in case of error this shouldn\'t be printed')