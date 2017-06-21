import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool('homopolymer_detection', True,
                  'Detect homopolymers')

FLAGS = flags.FLAGS






class basecallerInput(object):
    """
    Object containing input data
    
    """
    def __init__(self, config ,data, name=None):
        self.num_steps = config.num_steps
        self.input_data, self.targets = reader.construct_training_set(data, seq_length, name=name)

class basecallerModel(object):
    def __init__(self, is_training, config, input_):
        self._input  = input_

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                size, forget_bias=0.0, state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse)

        attn_cell = lstm_cell
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True
        )


class firstConfig(object):
    """
    first try at config object
    """
    num_steps = 1000
    num_layers = 1
    batch_size = 200
