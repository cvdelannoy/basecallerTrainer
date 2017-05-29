import numpy as np
import os
from cached_property import cached_property

# A class for MinION training reads, containing read name, raw signal, event
# durations and corresponding reference sequence.
# TODO: check fastq availability
# TODO: check basecalling


class TrainingRead:

    def __init__(self, hdf, readnb, refsam):
        """Initialize a new training read.

        Keyword arguments:
        hdf -- ref to an opened fast5 file
        readnb -- number of read contained in the fast5 file
        refbam -- bam file containing read mapping info
        """
        self.hdf = hdf
        self.readnb = readnb
        self.refsam = refsam
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
        Retrieve 5-mers as assigned by nanoraw, matching exactly to reference
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

    @property
    # TODO: finish
    def ref_sequence(self):
        """
        Retrieve reference sequence from sam-file
        """
        fastq_name = (self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Fastq']
                              [()].astype(str).split()[0])
        fastq_name = fastq_name[1:]  # remove at-sign of fastq
        sr = os.popen("grep -m 1 "
                      + fastq_name
                      + " " + self.refsam).read().split()
        # cigar = sr[6]
        seq = sr[10]
        return seq

    @cached_property
    def homopolymer_onehot(self):
        """
        Construct onehot vector, containing True when 5-mer is homopolymeric
        """
        if self.event_pattern_nanoraw is None:
            event_pattern = self.event_pattern_nanoraw
        else:
            event_pattern = self.event_pattern

        hpi = np.array([], dtype=int)
        for s in ['AAAAA', 'TTTTT', 'CCCCC', 'GGGGG']:
            hpi_cur = np.where(event_pattern == s)
            hpi = np.concatenate((hpi, hpi_cur[0]))
        hpi.sort()
        self._homopolymer_onehot_out = np.zeros(self.rawsignal.size, dtype=bool)
        self._homopolymer_onehot_out[hpi] = True
        return self._homopolymer_onehot_out
