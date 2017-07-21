import tensorflow as tf
import numpy as np


def npz_to_tf(npz, total_series_length=None, name=None):
    """Turn npz-files into tf-readable objects

    :param npz: npz-file storing raw data and labels 
    :return: two tensors, containing raw data and labels
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
