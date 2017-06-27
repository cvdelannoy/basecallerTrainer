import tensorflow as tf
import numpy as np
from BidirectionalRnn import BidirectionalRnn
# import cas.RWACell as rwa

# from cached_property import cached_property


class OrdinalBidirectionalRnn(BidirectionalRnn):
    def calculate_cost(self):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_hat,
                                                         labels=tf.cast(self.y, dtype=tf.float32))
        return tf.reduce_mean(tf.stack(losses))

    def evaluate_model(self):
        with tf.name_scope('Performance_assessment'):
            # Summary performance metrics
            y_hat = tf.cast(tf.round(tf.reduce_sum(self.y_hat, axis=-1)), dtype=tf.int32)
            y = tf.cast(tf.round(tf.reduce_sum(self.y, axis=-1)), dtype=tf.int32)
            # Overall accuracy
            accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_hat), dtype=tf.float32))

            # per class accuracy
            accuracy_list = []
            for i in range(self.num_classes):
                cm = tf.constant(i, shape=[self.batch_size, self.nb_steps], dtype=tf.int32)
                y_bin = tf.cast(tf.equal(y, cm), dtype=tf.int32)
                y_hat_bin = tf.cast(tf.equal(y_hat, cm), dtype=tf.int32)
                accuracy_class = tf.reduce_mean(tf.cast(tf.equal(y_bin, y_hat_bin), dtype=tf.float32))
                accuracy_list.append(accuracy_class)
                # TPR and TNR for HPs
                if i == (self.num_classes - 1):
                    TP = tf.count_nonzero(tf.multiply(y_bin, y_hat_bin))
                    TN = tf.count_nonzero(tf.multiply(y_bin - 1, y_hat_bin - 1))
                    TPR = TP / tf.count_nonzero(y_bin)
                    TNR = TN / tf.count_nonzero(y_bin - 1)

        with tf.name_scope('tensorboard_summaries'):
            tf.summary.scalar('TPR', TPR)
            tf.summary.scalar('TNR', TNR)
            tf.summary.scalar('accuracy', accuracy)
            for i in range(self.num_classes):
                tf.summary.scalar('accuracy_class%d' % i, accuracy_list[i])
        return tf.summary.merge_all()

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
