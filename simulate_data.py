#!/usr/bin/python
import numpy as np
import h5py
import random
import os
import argparse

import training_encodings
import readsim_model as rsm
from helper_functions import normalize_raw_signal

parser = argparse.ArgumentParser(description='Create training reads for RNN training in npz-format,'
                                             'with same characteristics as input read(s).')
parser.add_argument('real_reads', type=str, nargs='+',
                    help='Actual fast5-format MinION read, list of reads (or directory) on which event characteristics'
                         'will be based.')
parser.add_argument('-o', '--out-path', type=str, required=True,
                    help='Folder in which simulated reads should be saved. (Is created if not existing).')
parser.add_argument('-k', '--k-length', type=int, default=5, required=False,
                    help='k-mer length to which events are assigned.')
parser.add_argument('--nb-reads', type=int, default=1000, required=False,
                    help='Number of reads to simulate.')
parser.add_argument('--nb-bases', type=int, default=10000, required=False,
                    help='Average number of bases per simulated read.')
parser.add_argument('-s', '--add-noise', action='store_true', required=False,
                    help='Add normal noise to signal, with sd as encountered in real reads.')
parser.add_argument('--event-duration', type=int, default=8, required=False,
                    help='Average number of raw data points per event.')
parser.add_argument('--normalization', type=str, required=False, default='median',
                    help='Specify how the raw data should be normalized.')
parser.add_argument('--hp-prob', type=float, default=0.01, required=False,
                    help='Probability of including additional homopolymer events.')
parser.add_argument('--hp-length', type=int, default=10, required=False,
                    help='Length of added homopolymer events.')
parser.add_argument('--event-duration-sd', type=int, default=1, required=False,
                    help='standard deviation of normal distribution around event length')
parser.add_argument('--five-class', action='store_true',
                    help='Use five-class homopolymer encoding.(5th class is hp)')
parser.add_argument('--eight-class', action='store_true',
                    help='Use eight-class homopolymer encoding. (5th class is hp)')
parser.add_argument('--basecall-training-trimers', action='store_true',
                    help='create training set to recognize the middle trimer per kmer in 4 classes')
parser.add_argument('--basecall-training-pu-py', action='store_true',
                    help='create training set to recognize purines and pyrimidines')
args = parser.parse_args()

if not len(args.real_reads):
    raise ValueError('Supply at least one real_read.')
if len(args.real_reads) == 1 and os.path.isdir(args.real_reads[0]):
    args.real_reads = [args.real_reads[0] + rn for rn in os.listdir(args.real_reads[0])]

if args.out_path[-1] != '/':
    args.out_path += '/'

if not os.path.exists(args.out_path):
    os.mkdir(args.out_path)

bases = ['A', 'C', 'G', 'T']
hp_seqs = [i*args.k_length for i in bases]

# Find suitable read to construct model (i.e. representing all k-mers)
ri = 0
while ri <= len(args.real_reads):
    read = args.real_reads[ri]
    sim = rsm.readsim_model(k=args.k_length, hdf=h5py.File(read, 'r'))
    if sim.all_kmers_present is True:
        break
    ri += 1
if not sim.all_kmers_present:
    raise ValueError('None of provided reads represent all k-mer values, which is required for simulation.')


# Start read simulation
for nri in range(args.nb_reads):
    raw = np.empty(0, dtype=np.float32)
    encoded = np.empty(0, dtype=np.int32)
    base_labels = np.empty(0, dtype='str')
    cur_hp_length = 0; cur_encoded = 0
    sequence = ''.join([random.choice(bases) for _ in range(args.k_length-1)])
    for __ in range(args.nb_bases):
        if cur_hp_length > 0:
            cur_hp_length -= 1
        else:
            base = random.choice(bases)
            if random.random() < args.hp_prob:
                cur_hp_length = args.hp_length
        sequence += base
        cur_kmer = sequence[-args.k_length:]
        cur_event_duration = round(np.random.normal(args.event_duration, args.event_duration_sd))
        if cur_event_duration < 1:
            cur_event_duration = 1
        cur_raw = np.repeat(sim.raw_avg[cur_kmer], cur_event_duration)
        if args.add_noise:
            cur_raw += np.random.normal(cur_raw, sim.raw_stdv[cur_kmer])
        raw = np.concatenate((raw, cur_raw))
        base_labels = np.concatenate((base_labels, [cur_kmer] * cur_event_duration))
        if args.five_class:
            cur_encoded = training_encodings.hp_class_number(cur_kmer)
        elif args.basecall_training_trimers:
            cur_encoded = training_encodings.trimer_class_number(cur_kmer)
        elif args.basecall_training_pu_py:
            cur_encoded = training_encodings.pu_py_class_number(cur_kmer)
        else:
            cur_encoded = np.any(np.repeat(cur_kmer, len(hp_seqs)) == hp_seqs)
        encoded = np.concatenate((encoded, np.repeat(cur_encoded, cur_event_duration)))
    outName = args.out_path + os.path.basename(read)[:-6] + '_sim_onehot_' + str(nri) + '.npz'

    raw = normalize_raw_signal(raw, args.normalization)# Normalize signal
    np.savez(outName, raw=raw, onehot=encoded, base_labels=base_labels, sequence=sequence)
    if nri != 0 and nri % 100 == 0:
        print('%d reads simulated' % nri)
