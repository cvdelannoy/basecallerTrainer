from cached_property import cached_property
import h5py
import itertools
import numpy as np
import os
import math

# A class for capturing characteristics of a single MinION read, allowing generation of
# Raw signal training reads with similar characteristics.

class readsim_model:

    def __init__(self, k, hdf):
        self.hdf = hdf
        self.k = k
        self._raw_avg = None
        self._raw_sd = None

    @cached_property
    def kmers(self):
        return [''.join(i) for i in itertools.product(['A', 'C', 'T', 'G'], repeat=self.k)]

    @cached_property
    def raw(self):
        return self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Events']['mean']

    @cached_property
    def model_states(self):
        return self.hdf['Analyses/Basecall_1D_000/BaseCalled_template/Events']['model_state']

    @cached_property
    def all_kmers_present(self):
        for i in self.raw_avg:
            if math.isnan(self.raw_avg[i]):
                return False
        return True

    @cached_property
    def raw_avg(self):
        """
        Calculate average raw signal for every 5-mer
        :return: 
        """
        self.all_kmers_present = True
        nb_events = self.raw.size
        _raw_avg = {}
        for cur_kmer in self.kmers:
            cur_idx = self.model_states == cur_kmer.encode('UTF-8')
            if np.any(cur_idx):
                cur_avg = np.mean(self.raw[cur_idx])
            else:
                cur_avg = float('nan')
            _raw_avg[cur_kmer] = cur_avg
        return _raw_avg


# realreads_path = '/mnt/nexenta/lanno001/nobackup/readFiles/ecoliLoman/ecoliLoman/'
# read = os.listdir(realreads_path)[0]
# sim = readsim_model(k=5, hdf=h5py.File(realreads_path+read,'r'))
# tst = sim.raw_avg