import tensorflow as tf
import numpy as np
from BidirectionalRnn import BidirectionalRnn
# import cas.RWACell as rwa

# from cached_property import cached_property


class RegressionBidirectionalRnn(BidirectionalRnn):

    def set_y(self):
        return tf.placeholder(tf.float32, [self.batch_size, self.nb_steps])

    def reshape_labels(self, labels):
        """
        Push labels from classes 1-5 to equidistant points in range [0,1]
        """
        labels_reshape = np.zeros((labels.size - self.input_size + 1), dtype=int)
        i = 0
        while i + self.input_size < labels.size:
            labels_reshape[i] = labels[i + self.label_shift]
            i += 1
        labels_reshape = labels_reshape[:self.batch_size * self.nb_steps]
        labels_reshape = labels_reshape.reshape(self.batch_size, self.nb_steps)
        
        labels_reshape = labels_reshape * 0.2 - 0.1
        return labels_reshape

    @property
    def y_multiclass(self):
        return tf.cast((self.y + 0.1) / 0.2, dtype=tf.int32)

    @property
    def y_hat(self):
        return tf.cast(tf.round((tf.sigmoid(self.y_hat_logit) + 0.1) / 0.2), dtype=tf.int32)

    def calculate_cost(self):
        return tf.nn.l2_loss(self.y - tf.sigmoid(self.y_hat_logit))
