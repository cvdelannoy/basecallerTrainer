import tensorflow as tf
import numpy as np
from BidirectionalRnn import BidirectionalRnn
# import cas.RWACell as rwa

# from cached_property import cached_property


class ClassificationBidirectionalRnn(BidirectionalRnn):

    def set_y(self):
        return tf.placeholder(tf.int32, [self.batch_size, self.nb_steps])

    @property
    def y_multiclass(self):
        return self.y

    @property
    def y_hat(self):
        y_hat = tf.cast(tf.argmax(tf.nn.softmax(self.y_hat_logit, dim=-1), axis=-1), dtype=tf.int32)
        return y_hat

    def calculate_cost(self):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_hat_logit,
                                                                labels=self.y)
        return tf.reduce_mean(tf.stack(losses))

    def reshape_labels(self, labels):
        labels_reshape = np.zeros((labels.size - self.input_size + 1), dtype=int)
        i = 0
        while i + self.input_size < labels.size:
            labels_reshape[i] = labels[i + self.label_shift]
            i += 1
        labels_reshape = labels_reshape[:self.batch_size * self.nb_steps]
        labels_reshape = labels_reshape.reshape(self.batch_size, self.nb_steps) - 1  # <- NOTE: from [1-5] to [0-4]
        return labels_reshape
