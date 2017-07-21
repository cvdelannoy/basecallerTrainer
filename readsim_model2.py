from cached_property import cached_property
import itertools
import numpy as np
from math import isnan

# A class for capturing characteristics of a single MinION read, allowing generation of
# Raw signal training reads with similar characteristics.


class readsim_model:

    def __init__(self, condensed_events, k):
        self._raw_avg = None
        self._raw_stdv = None

        self.condensed_events = condensed_events
        self.k = k
        self.raw_avg = None

    @cached_property
    def kmers(self):
        return [''.join(i) for i in itertools.product(['A', 'C', 'T', 'G'], repeat=self.k)]

    @property
    def all_kmers_present(self):
        for i in self.raw_avg:
            if isnan(self.raw_avg[i]):
                return False
        return True

    @property
    def raw_avg(self):
        return self._raw_avg

    @property
    def raw_stdv(self):
        return self._raw_stdv

    @raw_avg.setter
    def raw_avg(self, _):
        """
        Calculate average raw signal for every 5-mer
        """
        raw_avg = {}
        raw_stdv = {}
        for cur_kmer in self.kmers:
            cur_raw = [cr for kmer, _, cr in self.condensed_events if kmer == cur_kmer]
            if len(cur_raw) == 0:
                raw_avg[cur_kmer] = float('nan')
                raw_stdv[cur_kmer] = float('nan')
            else:
                cur_raw = [point for sublist in cur_raw for point in sublist]  # flatten list
                raw_avg[cur_kmer] = np.mean(cur_raw)
                raw_stdv[cur_kmer] = np.std(cur_raw)
        self._raw_avg = raw_avg
        self._raw_stdv = raw_stdv
