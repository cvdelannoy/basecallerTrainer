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
                 dropout_keep_prob,
                 adaptive_positive_weighting
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
        self.adaptive_positive_weighting = adaptive_positive_weighting

        # (Partially) derived properties
        read_length_extended = (read_length - input_size + 1) * input_size
        self.nb_steps = read_length_extended // input_size // batch_size  # NOTE: floored division
        self.label_shift = (input_size - 1) // 2
        self.w = tf.Variable(tf.random_normal([2 * layer_size, num_classes]))
        self.b = tf.Variable(np.zeros([num_classes], dtype=np.float32))
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.nb_steps, self.input_size])
        self.is_target_kmer = tf.placeholder(tf.bool, [self.batch_size, self.nb_steps])
        self.y = self.set_y()
        # self.y = tf.placeholder(tf.int32, [self.batch_size, self.nb_steps, self.num_classes])
        self.y_hat_logit = self.construct_brnn()  # NOTE: NO SIGMOID YET --> range (-inf,inf)

        self.TPR = None
        self.TNR = None
        self.PPV = None
        # self.P = None  # for debugging purposes, remove below this afterward
        # self.N = None
        # self.TP = None
        # self.TN = None

        self.loss = self.calculate_cost()
        self.tb_summary = self.construct_evaluation('all_metrics')
        self.tb_roc = self.construct_evaluation('roc')

        self.session = None
        self.name_optimizer = name_optimizer
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
        return tf.squeeze(tf.stack(y_hat, axis=1))  # squeeze to remove dimensions of 1

    @optimizer.setter
    def optimizer(self, name_optimizer):
        # self.loss = self.calculate_cost()
        if name_optimizer == 'adam':
            # epsilon parameter improves numerical stability(?)
            self._optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.1).minimize(self.loss)
            # self._optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        elif name_optimizer == 'adagrad':
            self._optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
        elif name_optimizer == 'adadelta':
            self._optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss)
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

    def mark_target_kmers(self, kmers, target_kmers):
        if kmers is not None and target_kmers is not None:
            truncated_kmers = kmers[self.label_shift:]
            is_target_kmer = np.array([1 if km in target_kmers else 0 for km in truncated_kmers])
            is_target_kmer = is_target_kmer[:self.batch_size * self.nb_steps]
            is_target_kmer = is_target_kmer.reshape(self.batch_size, self.nb_steps)
        else:
            is_target_kmer = np.ones([self.batch_size, self.nb_steps])
        return is_target_kmer

    def initialize_model(self, params):
        self.session = tf.Session()
        if params is None:
            print("Initializing model with random parameters.")
            self.session.run(tf.global_variables_initializer())
        else:
            print("Loading model parameters from %s" % params)
            tf.train.Saver(tf.global_variables()).restore(self.session, params)

    def train_model(self, raw, labels, kmers=None, target_kmers=None):
        raw_reshape = self.reshape_raw(raw)
        labels_reshape = self.reshape_labels(labels)
        is_target_kmer = self.mark_target_kmers(kmers, target_kmers)
        self.session.run(self.optimizer, feed_dict={
            self.x: raw_reshape,
            self.y: labels_reshape,
            self.is_target_kmer: is_target_kmer
        })

    def evaluate_model(self, raw, labels, kmers=None, target_kmers=None):
        raw_reshape = self.reshape_raw(raw)
        labels_reshape = self.reshape_labels(labels)
        # If k-mers are supplied, use them to filter out predicted positives outside target kmers
        is_target_kmer = self.mark_target_kmers(kmers, target_kmers)
        tb_summary, loss, y_hat, TPR, TNR, PPV = self.session.run([self.tb_summary, self.loss, self.y_hat,
                                                                   self.TPR, self.TNR, self.PPV], feed_dict={
            self.x: raw_reshape,
            self.y: labels_reshape,
            self.is_target_kmer: is_target_kmer
            })
        return tb_summary, loss, y_hat, TPR, TNR, PPV

    def predict(self, raw):
        raw_reshape = self.reshape_raw(raw)
        y_hat = self.session.run(self.y_hat, feed_dict={
            self.x: raw_reshape
        })
        padding = np.repeat(np.NaN, self.label_shift)
        y_hat = y_hat.reshape(-1)
        # if self.perform_sanity_check:
        #     y_hat_s1 = np.roll(y_hat, -1)
        #     non_sequential_index = np.argwhere((y_hat - y_hat_s1) > 1)
        # Add padding, revert from batches to 1D
        return np.concatenate((padding, y_hat, padding))

    def construct_evaluation(self, mode):
        with tf.name_scope('Performance_assessment'):
            # Overall accuracy
            accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_multiclass, self.y_hat), dtype=tf.float32))

            # per class accuracy
            accuracy_list = []
            for i in range(1,self.num_classes+1):
                cm = tf.constant(i, shape=[self.batch_size, self.nb_steps], dtype=tf.int32)
                y_bin = tf.cast(tf.equal(self.y_multiclass, cm), dtype=tf.int32)
                y_hat_bin = tf.cast(tf.equal(self.y_hat, cm), dtype=tf.int32)
                accuracy_class = tf.reduce_mean(tf.cast(tf.equal(y_bin, y_hat_bin), dtype=tf.float32))
                accuracy_list.append(accuracy_class)
            # TPR and TNR for HPs
                # if i == (self.num_classes - 1):

            # pos_examples_mask = tf.not_equal(tf.reduce_sum(y_bin, axis=1), 0)
            # y_bin_masked = tf.boolean_mask(y_bin, pos_examples_mask)
            # y_hat_bin_masked = tf.boolean_mask(y_hat_bin, pos_examples_mask)

            TP = tf.count_nonzero(tf.multiply(y_bin, y_hat_bin))
            TN = tf.count_nonzero(tf.multiply(y_bin - 1, y_hat_bin - 1))
            P = tf.count_nonzero(y_bin)
            N = tf.count_nonzero(y_bin - 1)
            self.TPR = TP / P
            self.TNR = TN / N
            self.PPV = TP / tf.count_nonzero(y_hat_bin)


            # TP = tf.count_nonzero(tf.multiply(y_bin_masked, y_hat_bin_masked))
            # TN = tf.count_nonzero(tf.multiply(y_bin_masked - 1, y_hat_bin_masked - 1))
            # P = tf.count_nonzero(y_bin_masked)
            # N = tf.count_nonzero(y_bin_masked - 1)
            # self.TPR = TP / P
            # self.TNR = TN / N
            # self.PPV = TP / tf.count_nonzero(y_hat_bin_masked)

        with tf.name_scope('tensorboard_summaries'):
            if mode in ['all_metrics']:
                tf.summary.scalar('TPR', self.TPR)
                tf.summary.scalar('TNR', self.TNR)
                tf.summary.scalar('PPV', self.PPV)
                # roc_summary = tf.Summary()
                # roc_summary.value.add(tag='ROC', simple_value=self.TPR)
            if mode in ['all_metrics']:
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('accuracy', accuracy)
                for i in range(self.num_classes):
                    tf.summary.scalar('accuracy_class%d' % (i+1), accuracy_list[i])
        return tf.summary.merge_all()

    @abc.abstractmethod
    def calculate_cost(self):
        """
        Calculate cost using cost function defined in subclass 
        """
        raise ValueError('cost function not implemented for class')

    # @abc.abstractmethod
    # def construct_evaluation(self):
    #     """
    #     Define evaluation metrics and define how to calculate them
    #     """
    #     raise ValueError('Evaluation metrics not defined for class')

    @abc.abstractmethod
    def reshape_labels(self, labels):
        """
        Reshape data into form accepted by cost function
        """
        raise ValueError('Label reshape function not implemented for class')

    @abc.abstractmethod
    def set_y(self):
        raise ValueError('y not defined in class')

    @property
    @abc.abstractmethod
    def y_hat(self):
        raise ValueError('conversion of NN output to prediction not implemented for class')

    @property
    @abc.abstractmethod
    def y_multiclass(self):
        raise ValueError('y_multiclass not defined in class')
