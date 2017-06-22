import tensorflow as tf
import numpy as np
import os
import shutil
import time

from bokeh.models import ColumnDataSource, CategoricalColorMapper, LabelSet, Range1d
from bokeh.plotting import figure, save, output_file
from bokeh.io import show
from math import pi


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


def plot_timeseries(raw, num_classes, label_shift, y_hat, batch_size, nb_steps, base_labels):
    ts_plot = figure(title='Classified time series')
    ts_plot.grid.grid_line_alpha = 0.3
    ts_plot.xaxis.axis_label = 'nb events'
    ts_plot.yaxis.axis_label = 'current signal'
    y_range = raw.max() - raw.min()
    colors = ['#ffffff', '#fdcc8a', '#fc8d59', '#e34a33', '#b30000']
    col_mapper = CategoricalColorMapper(factors=list(range(num_classes)), palette=colors)
    source = ColumnDataSource(dict(
        raw=raw[(label_shift - 1):][:y_hat.size],
        event=list(range(batch_size * nb_steps)),
        cat=y_hat[0, :],
        cat_height=np.repeat(np.mean(raw[(label_shift - 1):][:y_hat.size]), batch_size * nb_steps),
        base_labels=base_labels[(label_shift - 1):][:y_hat.size]
    ))
    ts_plot.rect(x='event', y='cat_height', width=1, height=y_range, source=source,
                 fill_color={
                     'field': 'cat',
                     'transform': col_mapper
                 },
                 line_color=None)
    base_labels_LabelSet = LabelSet(x='event', y='cat_height',
                                    y_offset=-y_range, angle=-0.5 * pi,
                                    text='base_labels', text_baseline='middle',
                                    source=source)
    ts_plot.add_layout(base_labels_LabelSet)
    ts_plot.scatter(x='event', y='raw', source=source)
    ts_plot.plot_width = 10000
    ts_plot.plot_height = 500
    ts_plot.x_range = Range1d(0, 500)
    return ts_plot