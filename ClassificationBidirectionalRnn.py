import tensorflow as tf
import numpy as np
from BidirectionalRnn import BidirectionalRnn
# import cas.RWACell as rwa

# from cached_property import cached_property


class ClassificationBidirectionalRnn(BidirectionalRnn):
    def calculate_cost(self):
        y_unstacked =  tf.unstack(self.y, self.nb_steps, 1)
        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pr, labels=tf.cast(yi, dtype=tf.float32)) for pr, yi in zip(self.y_hat, y_unstacked)]
        return tf.reduce_mean(tf.stack(losses))

    def reshape_labels(self, labels):
        labels_reshape = np.zeros((labels.size - self.input_size + 1), dtype=int)
        i = 0
        while i + self.input_size < labels.size:
            labels_reshape[i] = labels[i + self.label_shift]
            i += 1
        labels_reshape = labels_reshape[:self.batch_size * self.nb_steps]
        labels_reshape = labels_reshape.reshape(self.batch_size, self.nb_steps)
        return labels_reshape
