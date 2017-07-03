#!/usr/bin/python
import numpy as np
import h5py
import random
import os
import argparse
import re

import readsim_model as rsm

parser = argparse.ArgumentParser(description='Create training reads for RNN training in npz-format,'
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
parser.add_argument('-s', '--signal-sd', type=float, default=1.0, required=False,
                    help='standard deviation of noise added to signal')
parser.add_argument('--event-duration', type=int, default=8, required=False,
                    help='Average number of raw data points per event.')
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
parser.add_argument('--basecall-training1', action='store_true',
                    help='create training set to recognize the middle trimer per kmer in 4 classes')
args = parser.parse_args()

if not len(args.real_reads):
    raise ValueError('Supply at least one real_read.')

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

def hp_class_number(kmer):
    k_length = len(kmer)
    class_list = []
    for base in [kmer[0], kmer[-1]]:
        pat = re.compile(base)
        pat_index = [m.start(0) for m in pat.finditer(kmer)]
        lst = [i in pat_index for i in range(k_length)]
        ccf = 0; ccr = 0; boolf=True; boolr=True
        for i in range(k_length):
            if not lst[i]:
                boolf = False  # If series of trues stops in fwd direction, stop adding
            if not lst[-i-1]:
                boolr = False  # If series of trues stops in bwd direction, stop adding
            if not boolf and not boolr:
                break  # If both series are discontinued, stop iterating
            ccf += boolf; ccr += boolr
        class_list += [ccf, ccr]
    return max(class_list + [1])  # return Nb in range 1( = no dimer at start) - k( = homopolymer)

cl1 = ['GGT','GGA', 'AGT','GGG','AGG','GAT','AGA','GAG','GAA','CGT','CGA','AAT','TGA','CGG','AAG','TGT']
cl2 = ['GGC','AAA','GAC','CAT','CAG','AGC','TGG','TAT','CAA','TAG','AAC','CGC','TAA','TGC','CAC','TAC']
cl3 = ['GCT', 'CCT', 'TCT', 'ACT','CCG','TTT','GTT','GCG','TCG','CTT','GCA','ACG','CCA','TCA','ATT','ACA']
cl4 = ['CCC', 'TTG', 'TCC', 'GTA','TTA','GTG','GCC','CTG','ACC', 'CTA','ATG','ATA','TTC','GTC','CTC','ATC']
def trimer_class_number(kmer):
    mid = len(kmer)//2 + 1
    trimer = kmer[mid-2:mid+1]
    if trimer in cl1:
        return 1
    if trimer in cl2:
        return 2
    if trimer in cl3:
        return 3
    if trimer in cl4:
        return 4
    raise ValueError('trimer not recognized.')


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
        raw = np.concatenate((raw, cur_raw))
        base_labels = np.concatenate((base_labels, [cur_kmer] * cur_event_duration))
        if args.five_class:
            cur_encoded = hp_class_number(cur_kmer)
        elif args.basecall_training1:
            cur_encoded = trimer_class_number(cur_kmer)
        else:
            cur_encoded = np.any(np.repeat(cur_kmer, len(hp_seqs)) == hp_seqs)
        encoded = np.concatenate((encoded, np.repeat(cur_encoded, cur_event_duration)))
    outName = args.out_path + os.path.basename(read)[:-6] + '_sim_onehot_' + str(nri) + '.npz'
    # sequence = np.array(list(sequence),dtype='str')
    raw += np.random.normal(0, args.signal_sd, size=raw.size)  # Add noise
    np.savez(outName, raw=raw, onehot=encoded, base_labels=base_labels, sequence=sequence)
    if nri != 0 and nri % 100 == 0:
        print('%d reads simulated' % nri)
