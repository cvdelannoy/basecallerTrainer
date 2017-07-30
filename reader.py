import numpy as np
import h5py
import re


def npz_to_tf(npz, total_series_length=None, name=None):
    """Turn npz-files into tf-readable objects
    :param npz: npz-file storing raw data and labels 

    """
    npz_read = np.load(npz)

    # If read length is smaller than set series length, discard
    if total_series_length is not None and npz_read['raw'].size < total_series_length:
        return None, None, None

    raw = npz_read['raw'][0:(total_series_length)]
    onehot = npz_read['onehot'][0:(total_series_length)]
    sequence = npz_read['base_labels'][0:(total_series_length)]

    npz_read.close()
    return raw, onehot, sequence

def fast5_to_tf(fast5):
    """
    Extract raw data and base labels from fast5-file, into numpy arrays 
    """
    readnb = int(re.search("(?<=read)\d+", fast5).group())
    with h5py.File(fast5, 'r') as hdf:
        first_sample = hdf['Analyses/Segmentation_000/Summary/segmentation'].attrs["first_sample_template"]
        raw = hdf['Raw/Reads/Read_%d/Signal' % readnb][()]
        raw = raw[first_sample:]
        base_labels = hdf['Analyses/Basecall_1D_000/BaseCalled_template/Events']['model_state']

    return raw, base_labels
