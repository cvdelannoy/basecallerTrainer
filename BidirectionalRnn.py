import tensorflow as tf
import numpy as np
import abc

# import cas.RWACell as rwa

# from cached_property import cached_property


class BidirectionalRnn(object):
    """ A model class for a tensorflow bidirectional recurrent neural network.


    """

    def __init__(self,
                 batch_size,
                 input_size,
                 num_layers,
                 read_length,
                 cell_type,
                 layer_size,
                 name_optimizer,
                 num_classes,
                 learning_rate,
                 dropout_keep_prob
                 ):
        self.read_length = read_length
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.layer_size = layer_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob

        # (Partially) derived properties
        read_length_extended = (read_length - input_size + 1) * input_size
        self.nb_steps = read_length_extended // input_size // batch_size  # NOTE: floored division
        self.label_shift = (input_size - 1) // 2
        self.w = tf.Variable(tf.random_normal([2 * layer_size, num_classes]))
        self.b = tf.Variable(np.zeros([num_classes], dtype=np.float32))
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.nb_steps, self.input_size])
        self.y = tf.placeholder(tf.int32, [self.batch_size, self.nb_steps, self.num_classes])
        # self.y = tf.unstack(self.y_placeholder, self.nb_steps, 1)  # to sub
        # self.x = tf.unstack(self.x_placeholder, self.nb_steps, 1)  # to sub
        self.y_hat_logit = self.construct_brnn()  # NOTE: NO SIGMOID YET!
        self.tb_summary = self.construct_evaluation()
        # self.tb_summary = None

        self.session = None
        self.optimizer = name_optimizer
        # self._optimizer = None

    # @cached_property
    # def y(self):
    #     return tf.unstack(self.y_placeholder, self.nb_steps, 1)
    #
    # @cached_property
    # def x(self):
    #     return tf.unstack(self.x_placeholder, self.nb_steps, 1)

    @property
    def base_cell(self):
        if self.cell_type == 'GRU':
            return tf.contrib.rnn.GRUCell(num_units=self.layer_size)
        if self.cell_type == 'LSTM':
            return tf.contrib.rnn.BasicLSTMCell(num_units=self.layer_size, forget_bias=1.0)
        # if self.cell_type == 'RWA':
        #     return rwa.RWACell(num_units=self.layer_size, decay_rate=0.0)
        return ValueError('Invalid cell_type given.')

    @property
    def optimizer(self):
        return self._optimizer

    def construct_brnn(self):
        """
        Method for the definition of the blstm that is to be trained/used
        """
        fwd_cell_list = []
        for fl in range(self.num_layers):
            with tf.variable_scope('forward%d' % fl, reuse=True):
                fwd_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.base_cell,
                                                                       output_keep_prob=self.dropout_keep_prob)
            fwd_cell_list.append(fwd_cell)
        fwd_multicell = tf.contrib.rnn.MultiRNNCell(fwd_cell_list)

        bwd_cell_list = []
        for bl in range(self.num_layers):
            with tf.variable_scope('backward%d' % bl, reuse=True):
                bwd_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.base_cell,
                                                                       output_keep_prob=self.dropout_keep_prob)
            bwd_cell_list.append(bwd_cell)
        bwd_multicell = tf.contrib.rnn.MultiRNNCell(bwd_cell_list)
        brnn_out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=fwd_multicell, cell_bw=bwd_multicell,
                                                                 inputs=tf.unstack(self.x, self.nb_steps, 1),
                                                                 dtype=tf.float32)
        y_hat = [tf.matmul(bo, self.w) + self.b for bo in brnn_out]
        return tf.stack(y_hat, axis=1)

    @optimizer.setter
    def optimizer(self, name_optimizer):
        loss = self.calculate_cost()
        if name_optimizer == 'adam':
            self._optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        else:
            raise ValueError('Given optimizer name %s not recognized' % name_optimizer)

    def reshape_raw(self, raw):
        """
        Reshape raw data vector into batch_size x nb_steps x input_size numpy matrix
        :param raw: Raw data as found in .npz-file, 1D-vector 
        :return: matrix of size batch_size x nb_steps x input_size
        """
        read_length_extended = (raw.size - self.input_size + 1) * self.input_size
        raw_reshape = np.zeros(read_length_extended)
        i = 0
        while i + self.input_size < raw.size:
            cur_row = raw[i:i + self.input_size]
            raw_reshape[i * self.input_size:i * self.input_size + self.input_size] = cur_row
            i += 1
        raw_reshape = raw_reshape[:self.nb_steps * self.input_size * self.batch_size]
        raw_reshape = raw_reshape.reshape(self.batch_size, self.nb_steps, self.input_size)
        return raw_reshape

    def initialize_model(self, params):
        self.session = tf.Session()
        if params is None:
            print("Initializing model with random parameters.")
            self.session.run(tf.global_variables_initializer())
        else:
            print("Loading model parameters from %s" % params)
            tf.train.Saver(tf.global_variables()).restore(self.session, params)

    def train_model(self, raw, labels):
        raw_reshape = self.reshape_raw(raw)
        labels_reshape = self.reshape_labels(labels)
        self.session.run(self.optimizer, feed_dict={
            self.x: raw_reshape,
            self.y: labels_reshape
        })

    def evalulate_model(self, raw, labels):
        raw_reshape = self.reshape_raw(raw)
        labels_reshape = self.reshape_labels(labels)
        tb_summary = self.session.run(self.tb_summary, feed_dict={
            self.x: raw_reshape,
            self.y: labels_reshape
        })
        return tb_summary

    def predict(self, raw):
        raw_reshape = self.reshape_raw(raw)
        y_hat = self.session.run(self.y_hat, feed_dict={
            self.x: raw_reshape
        })
        return y_hat

    @abc.abstractmethod
    def calculate_cost(self):
        """
        Calculate cost using cost function defined in subclass 
        """
        return ValueError('cost function not implemented for class')

    @abc.abstractmethod
    def construct_evaluation(self):
        """
        Define evaluation metrics and define how to calculate them 
        """
        return ValueError('Evaluation metrics not defined for class')

    @abc.abstractmethod
    def reshape_labels(self, labels):
        """
        Reshape data into form accepted by cost function
        """
        return ValueError('Label reshape function not implemented for class')

    @property
    @abc.abstractmethod
    def y_hat(self):
        return ValueError('conversion of NN output to prediction not implemented for class')