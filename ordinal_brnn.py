import tensorflow as tf
import numpy as np
import cas.RWACell as rwa

from cached_property import cached_property


class ordinal_brnn(object):
    """ A class for a tensorflow bidirectional recurrent neural network, for ordinal regression.
    
    
    """

    def __init__(self,
                 batch_size,
                 nb_steps,
                 input_size,
                 num_layers,
                 cell_type,
                 layer_size,
                 num_classes,
                 learning_rate,
                 dropout_keep_prob
                 ):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.input_size = input_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.layer_size = layer_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob
        # Derived properties
        self.label_shift = (input_size - 1)//2
        self.w = tf.Variable(tf.random_normal([2*layer_size, num_classes]))
        self.b = tf.Variable(np.zeros([num_classes], dtype=np.float32))

    @property
    def y(self):
        with tf.name_scope('input'):
            y_placeholder = tf.placeholder(tf.int64, [self.batch_size, self.nb_steps])
            y_out = tf.unstack(y_placeholder, self.nb_steps, 1)
        return y_out

    @property
    def x(self):
        with tf.name_scope('input'):
            x_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.nb_steps, self.input_size])
            x_out = tf.unstack(x_placeholder, self.nb_steps, 1)
        return x_out

    @property
    def brnn(self):
        """
        Method for the definition of the blstm that is to be trained/used
        :return: 
        """
        fwd_cell_list = []; bwd_cell_list=[]
        for fl in range(self.num_layers):
            with tf.variable_scope('forward%d' % fl, reuse=True):
                fwd_cell = self.base_cell
                fwd_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(fwd_cell, output_keep_prob=self.dropout_keep_prob)
            fwd_cell_list.append(fwd_cell)
        fwd_multicell = tf.contrib.rnn.MultiRNNCell(fwd_cell_list)
        for bl in range(self.num_layers):
            with tf.variable_scope('backward%d' % bl, reuse=True):
                bwd_cell = self.base_cell
                bwd_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(bwd_cell, output_keep_prob=self.dropout_keep_prob)
            bwd_cell_list.append(bwd_cell)
        bwd_multicell = tf.contrib.rnn.MultiRNNCell(bwd_cell_list)
        blstm_out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=fwd_multicell, cell_bw=bwd_multicell,
                                                                  inputs=self.x, dtype=tf.float32)
        return [tf.matmul(bo, self.w) + self.b for bo in blstm_out]


    def predict(self, x, y):
        with tf.Session as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run([y_hat_out], feed_dict={
                x_placeholder: x,
                y_placeholder: y
            })


    @cached_property
    def base_cell(self):
        """
        Set the cell_type for the brnn
        """
        if self.cell_type == 'GRU':
            return tf.contrib.rnn.GRUCell(num_units=self.layer_size)
        if self.cell_type == 'LSTM':
            return tf.contrib.rnn.BasicLSTMCell(num_units=self.layer_size, forget_bias=1.0)
        if self.cell_type == 'RWA':
            return rwa.RWACell(num_units=self.layer_size, decay_rate=0.0)
        return ValueError('Invalid cell_type given.')

