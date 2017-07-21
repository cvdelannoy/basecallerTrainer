import numpy as np
from helper_functions import normalize_raw_signal
from itertools import chain, repeat
from math import ceil
from random import choice, shuffle
from scipy.stats import mode
import training_encodings


# A class for nanoraw-corrected MinION training reads, containing raw signal, and the derivation of classes on which
# a neural network can train.

class TrainingRead:

    def __init__(self, hdf, readnb, normalization, use_nanoraw=False):
        """Initialize a new training read.

        """
        self._raw = None
        self.condensed_events = None
        self._event_length_list = None

        self.hdf = hdf
        self.readnb = readnb
        self.normalization = normalization
        self.use_nanoraw = use_nanoraw
        self.raw = None
        self.events = None

    def expand_sequence(self, sequence, length_list=None):
        """
        Expand a 1-event-per-item list to a one-raw-data-point-per-item list, using the event lengths derived from
        the basecaller. Uses event length list stored in object if none provided
        """
        if length_list is None:
            return list(chain.from_iterable(repeat(item, duration) for item, duration in zip(sequence,
                                                                                             self._event_length_list)))
        return list(chain.from_iterable(repeat(item, duration) for item, duration in zip(sequence,
                                                                                             length_list)))

    @property
    def clipped_bases_start(self):
        # Catches a version change!
        try:
            clipped_bases_start = self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs[
                'clipped_bases_start']
        except KeyError:
            clipped_bases_start = self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs[
                'trimmed_obs_start']
        return clipped_bases_start

    @property
    def clipped_bases_end(self):
        # Catches a version change!
        try:
            clipped_bases_end = self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs[
                'clipped_bases_end']
        except KeyError:
            clipped_bases_end = self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs[
                'trimmed_obs_end']
        return clipped_bases_end

    @property
    def raw(self):
        return self._raw

    @property
    def events(self):
        events, _, _ = zip(*self.condensed_events)
        return self.expand_sequence(events)

    @property
    def start_idx(self):
        _, start_idx, _ = zip(*self.condensed_events)
        return self.expand_sequence(start_idx)

    @property
    def event_length_list(self):
        return self._event_length_list

    @raw.setter
    def raw(self, _):
        if self.use_nanoraw:
            first_sample = self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'].attrs[
                'read_start_rel_to_raw']
        else:
            first_sample = self.hdf['Analyses/Segmentation_000/Summary/segmentation'].attrs["first_sample_template"]
        raw = self.hdf['Raw/Reads/Read_%d/Signal' % self.readnb][()]
        raw = raw[first_sample:]
        self._raw = normalize_raw_signal(raw, self.normalization)
        # self._raw = raw

    @events.setter
    def events(self, _):
        """
        Retrieve 5-mers and assign to corresponding raw data points.
        """
        event_lengths = self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Events']["length"]
        event_lengths *= 4000  # assumes 4000Hz sampling!
        event_lengths = np.round(event_lengths).astype(dtype=int)
        event_states = self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Events']["model_state"]
        event_states = event_states.astype(str)
        event_move = self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Events']["move"]
        event_move = event_move.astype(int)

        event_move[0] = 0  # Set first move to 0 in case it was not 0; used to set first-base index
        start_idx = 0
        # outputs
        event_list = [event_states[0]]  # list of k-mers assigned to events
        start_idx_list = [0]  # 0-based index of first base in event k-mer in fasta file
        event_length_list = []  # List of lengths of events in terms of raw data points
        event_raw_list = []  # List of lists containing raw data points per-event

        cur_event_length = event_lengths[0]
        temp_raw = list(self.raw)
        for n in range(event_move.size):
            if event_move[n] != 0:
                event_length_list.append(cur_event_length)
                event_raw_list.append(temp_raw[:cur_event_length])
                del temp_raw[:cur_event_length]
                cur_event_length = event_lengths[n]
                start_idx += event_move[n]
                event_list.append(event_states[n])
                start_idx_list.append(start_idx)
            else:
                cur_event_length += event_lengths[n]
        event_length_list.append(cur_event_length)  # Last event length
        event_raw_list.append(temp_raw[:cur_event_length]) # Last event raw data points
        del temp_raw[:cur_event_length]

        # Set clipped bases to 'NNNNN'
        # TODO: this could be prettier
        begin_clipped_end = len(event_list) - self.clipped_bases_end
        clip_idx = [sidx < self.clipped_bases_start or sidx + 5 > begin_clipped_end for sidx in start_idx_list]
        clip_idx = [i for i, x in enumerate(clip_idx) if x]
        for idx in clip_idx:
            event_list[idx] = 'NNNNN'

        self.condensed_events = list(zip(event_list,  # k-mers
                                          start_idx_list,  # index of first base in fasta
                                          event_raw_list))  # raw data points in event
        self._event_length_list = event_length_list

    def classify_events(self, encoding_type):
        """
        Return event labels as specified by encoding_type.
        """
        class_number_vec = np.vectorize(training_encodings.class_number)
        kmers, _, _ = zip(*self.condensed_events)
        try:
            class_numbers = class_number_vec(kmers, encoding_type)
        except IndexError:
            print('Index error, likely due to empty k-mer')
            return None
        return class_numbers

    def classify_deletion_affected_events(self):
        """
        Mark homopolymer using 5-class encoding, but only those that contain a deletion
        """
        # Find difference in starts of mapped and basecalled sequences
        first_sample_nanoraw = self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'].attrs[
            'read_start_rel_to_raw']
        first_sample_basecaller = self.hdf['Analyses/Segmentation_000/Summary/segmentation'].attrs[
            "first_sample_template"]
        offset_raw = first_sample_nanoraw - first_sample_basecaller
        if offset_raw < 0:
            raise ValueError('first_sample_nanoraw lower than first_sample_basecaller. This should not happen.')
        offset_check = 0
        offset_bases = 0
        while offset_check < offset_raw:
            offset_check += self._event_length_list[offset_bases]
            offset_bases += 1

        read_alignment = self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment']['read_alignment']
        kmers, _, _ = zip(*self.condensed_events)
        nb_deletions = 0
        hps = ['AAAAA', 'TTTTT', 'CCCCC', 'GGGGG']
        classes = [5, 4, 3, 2]
        class_number = np.ones(len(kmers), dtype=int)
        for rai in range(read_alignment.size):
            ei = rai + offset_bases - nb_deletions  # event index. Note: first base of event matches read alignment base
            if read_alignment[rai] == b'-':
                if kmers[ei] in hps:
                    for l in range(4):
                        if class_number[ei - l] == 1:
                            class_number[ei - l] = classes[l]
                        if class_number[ei + l] == 1:
                            class_number[ei + l] = classes[l]
                nb_deletions += 1
        # return self.expand_sequence(class_number)
        return class_number

    def oversample(self, class_number, read_model, fraction):
        # original_positives_idx = np.where(class_number == 5)
        events, start_idx, event_raw = zip(*self.condensed_events)
        condensed_events = list(zip(events, event_raw, class_number))
        op_and_lengths = [(op, l) for op, l in zip(condensed_events,
                                                   self._event_length_list) if op[2] == 5]
        original_positives, pos_length = zip(*op_and_lengths)
        pos_length = sum(pos_length)
        # original_positives = [condensed_events[idx] for idx in original_positives_idx]
        event_length_mode = mode(self._event_length_list)[0][0]

        # Determine how often the original positives have to be replicated to reach fraction
        # op_length = sum([self._event_length_list[idx] for idx in original_positives_idx])
        raw_length = len(self.raw)
        # nb_reps = ceil((fraction * raw_length - pos_length) /
        #                (pos_length - fraction * pos_length))

        # Add new positive events
        while pos_length / raw_length < fraction:
            for original_positive in original_positives:
                original_kmer = original_positive[0]

                # Find random place in read to insert new positive. Assert that:
                # - 9 subsequent events (to be replaced) are classified as 1
                # - new postive is not in or immediately adjacent unknown kmers
                # - previous k-mer does not end with same base as the new positive starts with
                # - next k-mer does not start with same base as the new positive ends with
                hp_base = original_kmer[0]
                classes_array = np.array([ce[2] for ce in condensed_events])
                candidate_insertion_places = [all(classes_array[c:c+9] == 1) for c in range(classes_array.size - 9)]
                candidate_insertion_places = np.where(candidate_insertion_places)[0]
                shuffle(candidate_insertion_places)
                class_at_idx_suitable = False
                for insertion_idx in candidate_insertion_places:
                    left_kmer = condensed_events[insertion_idx - 1][0]
                    right_kmer = condensed_events[insertion_idx+9][0]
                    class_at_idx_suitable = left_kmer != 'NNNNN'
                    class_at_idx_suitable *= right_kmer != 'NNNNN'
                    class_at_idx_suitable *= left_kmer[-1] != hp_base
                    class_at_idx_suitable *= right_kmer[0] != hp_base
                    if class_at_idx_suitable:
                        break
                if not class_at_idx_suitable:
                    raise ValueError('Unable to insert sufficient events to reach given fraction of positives')

                # form adjacent events
                replaced_length = [len(condensed_events[ci][1]) for ci in range(insertion_idx, insertion_idx + 9)]
                added_positive = [original_positive]
                for i in range(1, len(original_kmer)):
                    prev_kmer = left_kmer[-i:] + original_kmer[:-i]
                    next_kmer = original_kmer[i:] + right_kmer[:i]
                    prev_raw = np.random.normal(read_model.raw_avg[prev_kmer],
                                                read_model.raw_stdv[prev_kmer],
                                                size=replaced_length[4-i])

                    next_raw = np.random.normal(read_model.raw_avg[next_kmer],
                                                read_model.raw_stdv[next_kmer],
                                                size=replaced_length[4+i])
                    added_positive.insert(0, (prev_kmer, list(prev_raw), 5 - i))
                    added_positive.append((next_kmer, list(next_raw), 5 - i))
                condensed_events[insertion_idx:insertion_idx+9] = added_positive
                pos_length += len(original_positive[1])
                raw_length += len(original_positive[1]) - replaced_length[4]

        # Expand extended condensed_events list
        new_kmers, new_raw, new_class = zip(*condensed_events)
        new_event_lengths = [len(nr) for nr in new_raw]
        new_kmers = self.expand_sequence(new_kmers, new_event_lengths)
        new_class = self.expand_sequence(new_class, new_event_lengths)
        new_raw = [point for sublist in new_raw for point in sublist]

        return new_kmers, new_raw, new_class
