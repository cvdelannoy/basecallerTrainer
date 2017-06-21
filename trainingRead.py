import numpy as np
import re
from cached_property import cached_property

# A class for MinION training reads, containing read name, raw signal, event
# durations and corresponding reference sequence.
# TODO: check fastq availability
# TODO: check basecalling


class TrainingRead:

    def __init__(self, hdf, readnb, nanoraw_only, with_pos_only):
        """Initialize a new training read.

        Keyword arguments:
        hdf -- ref to an opened fast5 file
        readnb -- number of read contained in the fast5 file
        refbam -- bam file containing read mapping info
        """
        self.hdf = hdf
        self.readnb = readnb
        # self.refsam = refsam
        self.nanoraw_only = nanoraw_only
        self.with_pos_only = with_pos_only
        self._rawsignal_out = None
        self._event_pattern_out = None
        self._event_pattern_nanoraw_out = None
        self._homopolymer_onehot_out = None

    @cached_property
    def rawsignal(self):
        """
        Retrieve raw signal from read, store in object
        """
        self._rawsignal_out = self.hdf['Raw/Reads/Read_%d/Signal' % self.readnb][()]
        return self._rawsignal_out

    @cached_property
    def event_pattern(self):
        """
        Retrieve 5-mers corresponding to raw data points 
        NOTE: assumes sampling rate of 4000Hz!!
        """
        first_sample = self.hdf['Analyses/Segmentation_000/Summary/segmentation'].attrs["first_sample_template"]
        event_data = self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Events'][("length", "model_state")]
        event_data = np.array(event_data.tolist())

        self._event_pattern_out = np.zeros(self.rawsignal.size, dtype='<U5')
        self._event_pattern_out[0:first_sample] = "none"
        out_counter = first_sample
        for n in range(0, event_data.shape[0]):
            n_rawpoints = int(event_data[n, 0].astype(float) * 4000)
            self._event_pattern_out[out_counter:(out_counter+n_rawpoints+1)] = event_data[n, 1].astype(str)
            out_counter += n_rawpoints+1
        return self._event_pattern_out

    @cached_property
    def event_pattern_nanoraw(self):
        """
        Retrieve single nucleotides as assigned by nanoraw, convert to 5-mers
        and assign to corresponding raw data points.
        """
        try:
            event_data = (self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
                                  [("length", "base")])
        except KeyError:
            return None
        event_data = np.array(event_data.tolist())

        event_length = event_data[:, 0].astype(int)
        event_base = event_data[:, 1].astype(str)

        first_sample = self.hdf['Analyses/Segmentation_000/Summary/segmentation'].attrs["first_sample_template"]
        self._event_pattern_nanoraw_out = np.zeros(self.rawsignal.size, dtype='<U5')
        self._event_pattern_nanoraw_out[0:first_sample] = "none"

        fivemer = np.append(['N', 'N', 'N'], event_base[0:2])
        fivemer = ''.join(map(str, fivemer))
        out_counter = first_sample
        for n in range(0, event_length.size):
            fivemer = ''.join(map(str, np.append(fivemer[1:], event_base[n])))
            self._event_pattern_nanoraw_out[out_counter:out_counter+event_length[n]+1] = fivemer
            out_counter += event_length[n]+1
        if out_counter < self.rawsignal.size:
            self._event_pattern_nanoraw_out[out_counter:] = "none"
        return self._event_pattern_nanoraw_out

    @property
    def sequence(self):
        """
        Retrieve sequence derived by basecaller only (i.e. no reference)
        """
        return (self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Fastq']
                        [()].astype(str).split[2])

    # @property
    # # TODO: finish
    # def ref_sequence(self):
    #     """
    #     Retrieve reference sequence from sam-file
    #     """
    #     fastq_name = (self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Fastq']
    #                           [()].astype(str).split()[0])
    #     fastq_name = fastq_name[1:]  # remove at-sign of fastq
    #     sr = os.popen("grep -m 1 "
    #                   + fastq_name
    #                   + " " + self.refsam).read().split()
    #     # cigar = sr[6]
    #     seq = sr[10]
    #     return seq

    @cached_property
    def homopolymer_onehot(self):
        """
        If nanoraw was able to map, construct onehot vector containing True
        where 5-mer is homopolymeric
        """
        event_pattern = self._return_event_pattern()
        hpi = np.array([], dtype=int)
        for s in ['AAAAA', 'TTTTT', 'CCCCC', 'GGGGG']:
            hpi_cur = np.where(event_pattern == s)
            hpi = np.concatenate((hpi, hpi_cur[0]))
        if self.with_pos_only and hpi.size == 0:  # If no hits were found, return none
            return None
        hpi.sort()
        self._homopolymer_onehot_out = np.zeros(self.rawsignal.size, dtype=bool)
        self._homopolymer_onehot_out[hpi] = True
        return self._homopolymer_onehot_out

    def regex_onehot(self, pattern, min_length=5):
        """
        Construct a onehot vector denoting position at which the supplied regex
        pattern is found.
        Does not work on patterns longer or shorter than 5 nt's!
        :param pattern: 
        :return: 
        """
        if pattern[-2:] is not '\Z':
            raise ValueError('End regex pattern with a \Z to ensure that at mininimum,'
                             'an entire 5-mer is covered upon finding a hit.')
        if min_length < 5:
            raise ValueError('minimum match length should be 5 or more.')
        event_pattern = self._return_event_pattern
        r = re.compile(pattern)
        re_vector = np.vectorize(lambda x:bool(r.match(x)))
        match_indices = re_vector(event_pattern) # Locate initial matches
        all_match_indices = np.array([], dtype=int)
        for i in match_indices:
            if i not in all_match_indices:
                cur_match = event_pattern[i].tolist()
                cur_indices = np.array([], dtype=int)
                j = i
                while bool(r.match(cur_match)): # while match remains found, continue adding single bases
                    cur_indices = np.append(cur_indices, j)
                    j += 1
                    cur_match = cur_match + event_pattern[j].tolist()[-1]
                if cur_indices.size > min_length:
                    all_match_indices = np.append(all_match_indices, cur_indices)
        onehot_out = np.zeros(self.rawsignal.size, dtype=bool)
        onehot_out[all_match_indices] = True
        return onehot_out

    def _return_event_pattern(self):
        if self.event_pattern_nanoraw is None:
            if self.nanoraw_only:
                return None
            return self.event_pattern
        else:
            return self.event_pattern_nanoraw
