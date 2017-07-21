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
            y_hat_nlayer = tf.equal(y_hat[:, :, :n], cm)
            y_hat_nlayer = tf.floordiv(tf.reduce_sum(tf.cast(y_hat_nlayer, dtype=tf.int32), axis=-1), n)
            y_hat_nlayer = tf.expand_dims(y_hat_nlayer, axis=-1)
            y_hat_out = tf.concat([y_hat_out, y_hat_nlayer], axis=-1)
        y_hat_out = tf.reduce_sum(y_hat_out, axis=-1)

        # Account for supplied knowledge of kmers (target_kmer is all True if none is available)
        y_hat_is_target = tf.equal(y_hat_out, self.num_classes)
        y_hat_mask = tf.cast(tf.logical_and(y_hat_is_target, tf.logical_not(self.is_target_kmer)), dtype=tf.int32)
        y_hat_out = y_hat_out - tf.multiply(tf.subtract(y_hat_out, 1), y_hat_mask)
        return y_hat_out

    def calculate_cost(self):
        # losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_hat_logit,
        #                                                  labels=tf.cast(self.y, dtype=tf.float32))
        losses = []
        if self.adaptive_positive_weighting:
            y = tf.unstack(self.y, axis=0)
            y_hat_logit = tf.unstack(self.y_hat_logit, axis=0)
            for cl in range(self.num_classes):
                if cl in [0, 1, 2, 3]:
                    cur_loss = [
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_cur[:, cl], dtype=tf.float32),
                                                                 logits=y_hat_cur[:, cl]) for (y_cur,
                                                                                     y_hat_cur) in zip(y,
                                                                                                y_hat_logit)]
                else:
                    pos_examples = [tf.cast(tf.reduce_sum(y_cur[:, cl]),dtype=tf.float32) for y_cur in y]
                    pos_weights = [(tf.cast(self.nb_steps, dtype=tf.float32) - pe + 0.0001) / (pe + 0.0001) for pe in pos_examples]
                    cur_loss = [tf.nn.weighted_cross_entropy_with_logits(targets=tf.cast(y_cur[:, cl], dtype=tf.float32),
                                                                        logits=y_hat_cur[:, cl],
                                                                        pos_weight=pw) for (y_cur,
                                                                                            y_hat_cur,
                                                                                            pw) in zip(y,
                                                                                                       y_hat_logit,
                                                                                                       pos_weights)]
                losses += cur_loss
                # pos_examples = tf.reduce_sum(self.y[:,:,cl])
                # pos_weight = tf.cast((self.batch_size * self.nb_steps - pos_examples + 1) / (pos_examples + 1), dtype=tf.float32)
                # cur_loss = tf.nn.weighted_cross_entropy_with_logits(targets=tf.cast(self.y[:,:,cl], dtype=tf.float32),
                #                                                     logits=self.y_hat_logit[:,:,cl],
                #                                                     pos_weight=pos_weight)
                # losses += [cur_loss]
        else:
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.y, dtype=tf.float32),
                                                                 logits=self.y_hat_logit)

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
