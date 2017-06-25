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

        self.session = None


    @property
    def y_placeholder(self):
        with tf.name_scope('input'):
            y_placeholder = tf.placeholder(tf.int64, [self.batch_size, self.nb_steps])
        return tf.unstack(y_placeholder, self.nb_steps, 1)

    @property
    def x_placeholder(self):
        with tf.name_scope('input'):
            x_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.nb_steps, self.input_size])
        return tf.unstack(x_placeholder, self.nb_steps, 1)

    @property
    def base_cell(self):
        if self.cell_type == 'GRU':
            return tf.contrib.rnn.GRUCell(num_units=self.layer_size)
        if self.cell_type == 'LSTM':
            return tf.contrib.rnn.BasicLSTMCell(num_units=self.layer_size, forget_bias=1.0)
        if self.cell_type == 'RWA':
            return rwa.RWACell(num_units=self.layer_size, decay_rate=0.0)
        return ValueError('Invalid cell_type given.')

    @cached_property
    def brnn(self):
        """
        Method for the definition of the blstm that is to be trained/used
        :return: 
        """
        fwd_cell_list = []; bwd_cell_list=[]
        for fl in range(self.num_layers):
            with tf.variable_scope('forward%d' % fl, reuse=True):
                fwd_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.base_cell,
                                                                       output_keep_prob=self.dropout_keep_prob)
            fwd_cell_list.append(fwd_cell)
        fwd_multicell = tf.contrib.rnn.MultiRNNCell(fwd_cell_list)
        for bl in range(self.num_layers):
            with tf.variable_scope('backward%d' % bl, reuse=True):
                bwd_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.base_cell,
                                                                       output_keep_prob=self.dropout_keep_prob)
            bwd_cell_list.append(bwd_cell)
        bwd_multicell = tf.contrib.rnn.MultiRNNCell(bwd_cell_list)
        brnn_out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=fwd_multicell, cell_bw=bwd_multicell,
                                                            inputs=self.x_placeholder, dtype=tf.float32)
        y_hat = [tf.matmul(bo, self.w) + self.b for bo in brnn_out]
        return y_hat

    @cached_property
    def optimizer(self):
        losses = [tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pr, labels=tf.cast(yi, dtype=tf.float32)) for pr, yi in zip(self.brnn, self.y_placeholder)]
        mean_loss = tf.reduce_mean(tf.stack(losses))
        return tf.train.AdamOptimizer(self.learning_rate).minimize(mean_loss)

    def initialize_model(self, params):
        model = self.brnn
        self.session = tf.Session()
        if params is None:
            print("Initializing model with random parameters.")
            self.session.run(tf.global_variables_initializer())
        else:
            print("Loading model parameters from %s" % params)
            tf.train.Saver(tf.global_variables()).restore(self.session, params)


    def train_model(self, raw, labels):
        self.session.run(self.optimizer, feed_dict={
            x_placeholder: raw,
            y_placeholder: labels
        })


    def evaluate_performance(self, raw, labels, metrics):
        """
        
        :param raw: numpy object containing raw data
        :param labels:  numpy object containing labels
        :param metrics: list of model performance metrics
        :return: 
        """
        return self.session.run([metrics],feed_dict = {x_placeholder: raw,
                                                       y_placeholder: labels})

    def predict(self, raw):
        return self.session.run([y_hat],feed_dict = {x_placeholder: raw})

