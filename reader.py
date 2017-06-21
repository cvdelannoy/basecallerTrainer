import tensorflow as tf
import numpy as np

def npz_to_tf(npz, total_series_length, name=None):
    """Turn npz-files into tf-readable objects

    :param npz: npz-file storing raw data and labels 
    :return: two tensors, containing raw data and labels
    """
    npz_read = np.load(npz)

    # If read length is smaller than set series length, discard
    if npz_read['raw'].size < total_series_length:
        return None, None

    raw = npz_read['raw'][0:(total_series_length)]
    onehot = npz_read['onehot'][0:(total_series_length)]
    sequence = npz_read['base_labels'][0:(total_series_length)]

    npz_read.close()
    return raw, onehot, sequence


# def npz_to_tf(npz, batch_size, name=None):
#     """Turn npz-files into tf-readable objects
#
#     :param npz: npz-file storing raw data and labels
#     :return: two tensors, containing raw data and labels
#     """
#     npz_read = np.load(npz)
#     raw = npz_read['raw']
#     onehot = npz_read['onehot']
#     npz_read.close()
#
#     # If read length is too small to get at least one batch, discard
#     if raw.size < batch_size:
#         return None
#
#     with tf.name_scope(name, 'npz_to_tf', [raw, onehot, batch_size]):
#         # raw = tf.convert_to_tensor(raw, name="raw", dtype=tf.int16)
#         # data_len = tf.size(raw)
#         data_len = raw.size
#         batch_len = data_len // batch_size
#         raw = tf.reshape(raw[0: batch_size * batch_len],
#                           [batch_size, batch_len])
#         # epoch_size = (batch_len - 1) // num_steps
#         raw = tf.convert_to_tensor(raw, name="raw", dtype=tf.int16)
#         onehot = tf.convert_to_tensor(onehot, name='onehot', dtype=tf.bool)
#         onehot = tf.reshape(onehot[0: batch_size * batch_len],
#                          [batch_size, batch_len])
#
#         return raw, onehot
