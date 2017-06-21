import tensorflow as tf
import os
import shutil
import time


def set_logfolder(sess, parent_dir, batch_size, layer_size, learning_rate, pos_weight, epoch_index, cell_type):
    """
    Create a folder to store tensorflow metrics for tensorboard and set it up for a specific session.
    :param sess: 
    :param parent_dir: 
    :param batch_size: 
    :param input_size: 
    :param learning_rate: 
    :param pos_weight: 
    :param epoch_index: 
    :return: 
    """
    timedate = time.strftime('%y%m%d_%H%M%S')
    cur_tb_path = parent_dir + '%s_%s_bs%s_is%s_lr%s_pw%s_ep%s/' % (timedate, cell_type, batch_size, layer_size,
                                                                 learning_rate, pos_weight,
                                                                 epoch_index)
    if os.path.isdir(cur_tb_path):
        shutil.rmtree(cur_tb_path)
    os.mkdir(cur_tb_path)
    return tf.summary.FileWriter(cur_tb_path, sess.graph)
