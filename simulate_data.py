#!/usr/bin/python
import numpy as np
import h5py
import random
import os
import argparse
import re

import readsim_model as rsm

parser = argparse.ArgumentParser(description='Create training reads for homopolymer RNN training in npz-format,'
                                             'with same characteristics as input read(s).')
parser.add_argument('real_reads', type=str, nargs='+',
                    help='Actual fast5-format MinION read or list of reads on which event characteristics'
                         'will be based.')
parser.add_argument('-o', '--out-path', type=str, required=True,
                    help='Folder in which simulated reads should be saved. (Is created if not existing).')
parser.add_argument('-k', '--k-length', type=int, default=5, required=False,
                    help='k-mer length to which events are assigned.')
parser.add_argument('--nb-reads', type=int, default=1000, required=False,
                    help='Number of reads to simulate.')
parser.add_argument('--nb-bases', type=int, default=10000, required=False,
                    help='Average number of bases per simulated read.')
parser.add_argument('--event-duration', type=int, default=8, required=False,
                    help='Average number of raw data points per event.')
parser.add_argument('--hp-prob', type=float, default=0.01, required=False,
                    help='Probability of including additional homopolymer events.')
parser.add_argument('--hp-length', type=int, default=10, required=False,
                    help='Length of added homopolymer events.')
parser.add_argument('--multiclass', action='store_true',
                    help='Instead of onehot-encoding, use multiclass to also denote before and after events.')
args = parser.parse_args()

if not len(args.real_reads):
    raise ValueError('Supply at least one real_read.')

if args.out_path[-1] != '/':
    args.out_path += '/'

if not os.path.exists(args.out_path):
    os.mkdir(args.out_path)

# DEBUGGING ARGS
# poremodel_path = '~/.local/lib/python3.5/site-packages/albacore/data_versioned/'
# realreads_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman/'
# out_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman_simtr/'
#
# k_length = 5
# nb_reads = 1000
# nb_bases = 10000
# event_duration = 8
# hp_prob = 0.01
# hp_length = [10]
# noise = 0

def hp_class_number(base,kmer):
    k_length = len(kmer)
    pat = re.compile(base)
    pat_index = [m.start(0) for m in pat.finditer(kmer)]
    lst = [i in pat_index for i in range(k_length)]
    ccf = 0; ccr = 0; boolf=True; boolr=True
    for i in range(k_length):
        if not lst[i]:
            boolf = False  # If series of trues stops in fwd direction, stop adding
        if not lst[-i-1]:
            boolr = False # If series of trues stops in bwd direction, stop adding
        if not boolf and not boolr:
            break  # If both series are discontinued, stop iterating
        ccf += boolf; ccr += boolr
    cc = max([ccf, ccr]) - 1  # -1 as a single match to the base is not a start of a hompolymer stretch
    return(max([ccf, ccr, 1]) - 1)  # -1 as starting with base is not sign of hp

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
            if cur_encoded == 0:
                hp_base = base  # If currently coming from homopolymer region, do NOT change ref base!
            is_hp = random.random() < args.hp_prob
            if is_hp:
                cur_hp_length = args.hp_length
        sequence += base
        cur_kmer = sequence[-args.k_length:]
        cur_raw = np.repeat(sim.raw_avg[cur_kmer], args.event_duration)
        raw = np.concatenate((raw, cur_raw))
        base_labels = np.concatenate((base_labels, [cur_kmer] * args.event_duration))
        if args.multiclass:
            cur_encoded = hp_class_number(hp_base, cur_kmer)
        else:
            cur_encoded = np.any(np.repeat(cur_kmer, len(hp_seqs)) == hp_seqs)
        encoded = np.concatenate((encoded, np.repeat(cur_encoded, args.event_duration)))
    outName = args.out_path + os.path.basename(read)[:-6] + '_sim_onehot_'+ str(nri) + '.npz'
    # sequence = np.array(list(sequence),dtype='str')
    np.savez(outName, raw=raw, onehot=encoded, base_labels=base_labels, sequence=sequence)
    if nri != 0 and nri % 100 == 0:
        print('%d reads simulated' % nri)
