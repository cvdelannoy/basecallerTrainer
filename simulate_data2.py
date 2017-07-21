#!/usr/bin/python
import h5py
import numpy as np
import re
import os
import argparse
import random

from training_encodings import valid_encoding_types, class_number
import trainingRead as tr
import readsim_model2 as rs

parser = argparse.ArgumentParser(description='Convert MinION fast5-reads into simulated training reads,'
                                             'with similar characteristics as real reads. Optionally, add'
                                             'target event with given probability at each position.')
inputArg = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('-o', '--outFolder', type=str, required=True,
                    help='Folder in which results are stored')
parser.add_argument('-k', '--k-length', type=int, default=5, required=False,
                    help='k-mer length to which events are assigned.')
parser.add_argument('-n', '--normalization', type=str, required=False, default='median',
                    help='Specify how the raw data should be normalized.')
parser.add_argument('--nb-reads', type=int, default=1000, required=False,
                    help='Number of reads to simulate.')
parser.add_argument('--nb-bases', type=int, default=10000, required=False,
                    help='Average number of bases per simulated read.')
parser.add_argument('-c', '--encoding-type', type=str, required=False, default='hp_5class',
                    help='Specify which kind of classification to adhere to. Must be one of the following types: %s'
                    % ', '.join(valid_encoding_types))
parser.add_argument('--deletion-affected', action='store_true',
                    help='Only mark events that were affected by a deletion.')
parser.add_argument('--add-target-prob', type=float, default=0.01, required=False,
                    help='Probability of including additional target events.')
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

read_count = 0; success_count = 0
pattern = "(?<=read)\d+"

bases = ['A', 'C', 'G', 'T']
hps = [i*args.k_length for i in bases]

for read in reads:
    if not read_count % 10:
        print("%d reads processed, %d training files created" % (read_count,
                                                                 success_count))
    read_count += 1
    try:
        readnb = int(re.search(pattern, read).group())
        hdf = h5py.File(read, 'r')
        read_count += 1
        tr_cur = tr.TrainingRead(hdf, readnb, args.normalization)
        if args.deletion_affected:
            encoded = tr_cur.classify_deletion_affected_events()  # encoding: 1 per kmer!
        else:
            encoded = tr_cur.classify_events(args.encoding_type)  # encoding: 1 per kmer!

        if tr_cur.events is None or encoded is None or 5 not in encoded:
            continue

        # Construct read-sim model, containing current mean and stdv for every k-mer
        rs_cur = rs.readsim_model(tr_cur.condensed_events, k=5)
        if not rs_cur.all_kmers_present:
            continue

        con_kmers, _, con_raw = zip(*tr_cur.condensed_events)
        # If simulating deletion-affected reads, collect deletion-affected events
        if args.deletion_affected:
            deletion_affected_events = [(km, rw, cl) for km, rw, cl in zip(con_kmers, con_raw, encoded) if cl == 5]

        # Construct list of event lengths
        event_lengths = [l for l, cl, km in zip(tr_cur.event_length_list, encoded, con_kmers) if cl != 5 and km != 'NNNNN']

        # Construct the simulated read
        insert_target_event = False
        cur_encoded = 1
        cur_kmer = ''.join([random.choice(bases) for _ in range(args.k_length)])
        sim_condensed_events = []
        event_duration_list = []
        for __ in range(args.nb_bases):
            if not insert_target_event or insert_target_event and cur_kmer in hps:
                insert_target_event = False
                base = random.choice(bases)


            cur_kmer = cur_kmer[1:] + base

            if args.deletion_affected and insert_target_event and cur_kmer in hps:
                cur_condensed_event = del_affected_event
                event_duration_list.append(len(del_affected_event[1]))
            else:
                cur_event_duration = random.choice(event_lengths)
                event_duration_list.append(cur_event_duration)
                cur_raw = list(np.random.normal(loc=rs_cur.raw_avg[cur_kmer],
                                                scale=rs_cur.raw_stdv[cur_kmer],
                                                size=cur_event_duration))
                cur_encoded = class_number(cur_kmer, args.encoding_type)
                cur_condensed_event = (cur_kmer, cur_raw, cur_encoded)

            sim_condensed_events.append(cur_condensed_event)

            # Decide whether to start a new target event next iteration
            if not insert_target_event and random.random() < args.add_target_prob:
                insert_target_event = True
                if args.deletion_affected:
                    del_affected_event = random.choice(deletion_affected_events)
                    base = del_affected_event[0][0]

        # Expand condensed list and save
        base_labels, raw, encoded = zip(*sim_condensed_events)
        base_labels = tr_cur.expand_sequence(base_labels, length_list=event_duration_list)
        encoded = tr_cur.expand_sequence(encoded, length_list=event_duration_list)
        raw = [point for sublist in raw for point in sublist]

        outName = outFolder + os.path.basename(read)[:-6] + '_sim.npz'
        np.savez(outName,
                 raw=raw,
                 onehot=encoded,
                 base_labels=base_labels
                 )
        success_count += 1
        hdf.close()
    except KeyError:
        print('Key error')
        hdf.close()
        continue
    except IndexError:
        print('Index error')
        hdf.close()
        continue
    except ValueError:
        print('Value error')
        hdf.close()
        continue
