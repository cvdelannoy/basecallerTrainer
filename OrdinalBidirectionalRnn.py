import tensorflow as tf
import numpy as np
from BidirectionalRnn import BidirectionalRnn
# import cas.RWACell as rwa
# from cached_property import cached_property


class OrdinalBidirectionalRnn(BidirectionalRnn):

    def set_y(self):
        return tf.placeholder(tf.int32, [self.batch_size, self.nb_steps, self.num_classes])

    @property
    def y_multiclass(self):
        """
        Convert ordinal one-hot labels back to single-file classification labels 
        """
        y_hat = tf.cast(tf.round(tf.reduce_sum(self.y, axis=-1)), dtype=tf.int32)
        return y_hat

    @property
    def y_hat(self):
        y_hat = tf.cast(tf.round(tf.sigmoid(self.y_hat_logit)), dtype=tf.int32)
        y_hat_out = tf.ones([self.batch_size, self.nb_steps, 1], dtype=tf.int32)
        for n in range(2, self.num_classes + 1):
            cm = tf.ones([self.batch_size, self.nb_steps, n], dtype=tf.int32)
            y_hat_nlayer = tf.equal(y_hat[:,:,:n], cm)
            y_hat_nlayer = tf.floordiv(tf.reduce_sum(tf.cast(y_hat_nlayer, dtype=tf.int32), axis=-1), n)
            y_hat_nlayer = tf.expand_dims(y_hat_nlayer, axis=-1)
            y_hat_out = tf.concat([y_hat_out, y_hat_nlayer], axis=-1)
        y_hat_out = tf.reduce_sum(y_hat_out, axis=-1)
        return y_hat_out

    def calculate_cost(self):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_hat_logit,
                                                         labels=tf.cast(self.y, dtype=tf.float32))
        return tf.reduce_mean(tf.stack(losses))

    def reshape_labels(self, labels):
        labels_reshape = np.zeros((labels.size - self.input_size + 1, self.num_classes), dtype=int)
        i = 0
        while i + self.input_size < labels.size:
            cur_labels = np.zeros(self.num_classes, dtype=int)
            cur_labels[0:labels[i + self.label_shift]] = 1
            labels_reshape[i, :] = cur_labels
            i += 1
        labels_reshape = labels_reshape[:self.batch_size * self.nb_steps, :]
        labels_reshape = labels_reshape.reshape(self.batch_size, self.nb_steps, self.num_classes)
        return labels_reshape
