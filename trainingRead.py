import numpy as np
# import re
from helper_functions import normalize_raw_signal

import training_encodings


# A class for nanoraw-corrected MinION training reads, containing raw signal, called events as 5-mers and

class TrainingRead:

    def __init__(self, hdf, readnb, normalization, use_nanoraw=True):
        """Initialize a new training read.

        """
        self._raw = None
        self._events = None

        self.hdf = hdf
        self.readnb = readnb
        self.normalization = normalization
        self.use_nanoraw = use_nanoraw
        self.raw = None
        self.events = None


    @property
    def raw(self):
        return self._raw

    @property
    def events(self):
        return self._events

    @raw.setter
    def raw(self, _):
        if self.use_nanoraw:
            first_sample = self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'].attrs['read_start_rel_to_raw']
        else:
            first_sample = self.hdf['Analyses/Segmentation_000/Summary/segmentation'].attrs["first_sample_template"]
        raw = self.hdf['Raw/Reads/Read_%d/Signal' % self.readnb][()]
        raw = raw[first_sample:]
        self._raw = normalize_raw_signal(raw, self.normalization)

    @events.setter
    def events(self, _):
        """
        Retrieve 5-mers, or when using nanoraw data, get single nucleotides as assigned by nanoraw and convert to 5-mers
        and assign to corresponding raw data points.
        
        """
        # TODO: very ugly solution for nanoraw vs non-nanoraw
        if self.use_nanoraw:
            # If nanoraw succesfully mapped and corrected read, get lengths and individual bases
            event_data = (self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
                                  [("length", "base")])
            event_data = np.array(event_data.tolist())
            event_length = event_data[:, 0].astype(int)
            event_base = event_data[:, 1].astype(str)
            self._events = np.zeros(self.raw.size, dtype='<U5')
            # fivemer = ['N', 'N', 'N'] + event_base[0:2]
            fivemer = np.append(['N', 'N', 'N'], event_base[0:2])
            fivemer = ''.join(map(str, fivemer))
            out_counter = 0
            for n in range(0, event_length.size):
                fivemer = ''.join(map(str, np.append(fivemer[1:], event_base[n])))
                self._events[out_counter:out_counter + event_length[n] + 1] = fivemer
                out_counter += event_length[n] + 1
            if out_counter < self.raw.size:
                self._events[out_counter:] = "NNNNN"
        else:
            event_starts = self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Events'][("start")]
            event_starts -= event_starts[0]  # set start of read to 0 and convert to Nb of points
            event_starts *= 4000
            event_starts = np.round(event_starts).astype(dtype=int)
            event_lengths = self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Events'][("length")]
            event_lengths *= 4000
            event_lengths = np.round(event_lengths).astype(dtype=int)
            event_states = self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Events'][("model_state")]
            event_states = event_states.astype(str)
            self._events = np.zeros(self.raw.size, dtype='<U5')
            for n in range(event_starts.shape[0]-1):
                self._events[event_starts[n]:event_starts[n+1]] = event_states[n]
            end_last_event = event_starts[-1] + event_lengths[-1]
            if self.raw.size > end_last_event + 5:
                self._events[event_starts[-1]:end_last_event] = event_states[-1]
                self._events[end_last_event:] = 'NNNNN'
            else:
                self._events[event_starts[-1]:] = event_states[-1]
            # event_data = self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Events'][("length", "model_state")]
            # event_data = np.array(event_data.tolist())
            # self._events = np.zeros(self.raw.size, dtype='<U5')
            # out_counter = 0
            # for n in range(0, event_data.shape[0]):
            #     n_rawpoints = int(event_data[n, 0].astype(float) * 4000)
            #     self._events[out_counter:(out_counter + n_rawpoints + 1)] = event_data[n, 1].astype(str)
            #     out_counter += n_rawpoints + 1

    def classify_events(self, encoding_type):
        """
        Return event labels as specified by encoding_type.
        """
        class_number_vec = np.vectorize(training_encodings.class_number)
        return class_number_vec(self.events[2:-2], encoding_type)

    # @cached_property
    # def events(self):
    #     """
    #     Retrieve 5-mers corresponding to raw data points
    #     NOTE: assumes sampling rate of 4000Hz!!
    #     """
    #     first_sample = self.hdf['Analyses/Segmentation_000/Summary/segmentation'].attrs["first_sample_template"]
    #     event_data = self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Events'][("length", "model_state")]
    #     event_data = np.array(event_data.tolist())
    #
    #     self._event_pattern_out = np.zeros(self.raw_signal.size, dtype='<U5')
    #     self._event_pattern_out[0:first_sample] = "none"
    #     out_counter = first_sample
    #     for n in range(0, event_data.shape[0]):
    #         n_rawpoints = int(event_data[n, 0].astype(float) * 4000)
    #         self._event_pattern_out[out_counter:(out_counter+n_rawpoints+1)] = event_data[n, 1].astype(str)
    #         out_counter += n_rawpoints+1
    #     return self._event_pattern_out

    # @property
    # def sequence(self):
    #     """
    #     Retrieve sequence derived by basecaller only (i.e. no reference)
    #     """
    #     return (self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Fastq']
    #                     [()].astype(str).split[2])
    #
    #
    # @cached_property
    # def homopolymer_onehot(self):
    #     """
    #     DEPRECATED
    #     If nanoraw was able to map, construct onehot vector containing True
    #     where 5-mer is homopolymeric
    #     """
    #     event_pattern = self._return_event_pattern()
    #     hpi = np.array([], dtype=int)
    #     for s in ['AAAAA', 'TTTTT', 'CCCCC', 'GGGGG']:
    #         hpi_cur = np.where(event_pattern == s)
    #         hpi = np.concatenate((hpi, hpi_cur[0]))
    #     if self.with_pos_only and hpi.size == 0:  # If no hits were found, return none
    #         return None
    #     hpi.sort()
    #     self._homopolymer_onehot_out = np.zeros(self.raw.size, dtype=bool)
    #     self._homopolymer_onehot_out[hpi] = True
    #     return self._homopolymer_onehot_out
    #
    # def regex_onehot(self, pattern, min_length=5):
    #     """
    #     DEPRECATED
    #     Construct a onehot vector denoting position at which the supplied regex
    #     pattern is found.
    #     Does not work on patterns longer or shorter than 5 nt's!
    #     :param pattern:
    #     :return:
    #     """
    #     if pattern[-2:] is not '\Z':
    #         raise ValueError('End regex pattern with a \Z to ensure that at mininimum,'
    #                          'an entire 5-mer is covered upon finding a hit.')
    #     if min_length < 5:
    #         raise ValueError('minimum match length should be 5 or more.')
    #     event_pattern = self._return_event_pattern
    #     r = re.compile(pattern)
    #     re_vector = np.vectorize(lambda x:bool(r.match(x)))
    #     match_indices = re_vector(event_pattern) # Locate initial matches
    #     all_match_indices = np.array([], dtype=int)
    #     for i in match_indices:
    #         if i not in all_match_indices:
    #             cur_match = event_pattern[i].tolist()
    #             cur_indices = np.array([], dtype=int)
    #             j = i
    #             while bool(r.match(cur_match)): # while match remains found, continue adding single bases
    #                 cur_indices = np.append(cur_indices, j)
    #                 j += 1
    #                 cur_match = cur_match + event_pattern[j].tolist()[-1]
    #             if cur_indices.size > min_length:
    #                 all_match_indices = np.append(all_match_indices, cur_indices)
    #     onehot_out = np.zeros(self.raw_signal.size, dtype=bool)
    #     onehot_out[all_match_indices] = True
    #     return onehot_out
    #
    #
    # def _return_event_pattern(self):
    #     if self.event_pattern_nanoraw is None:
    #         if self.nanoraw_only:
    #             return None
    #         return self.event_pattern
    #     else:
    #         return self.event_pattern_nanoraw
