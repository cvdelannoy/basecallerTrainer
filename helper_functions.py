import tensorflow as tf
import numpy as np
import os
import shutil
import time

from bokeh.models import ColumnDataSource, CategoricalColorMapper, LabelSet, Range1d
from bokeh.plotting import figure

from math import pi


def set_logfolder(brnn_object, parent_dir, epoch_index):
    """
    Create a folder to store tensorflow metrics for tensorboard and set it up for a specific session.
    Returns a filewriter object, which can be used to write info to tensorboard.
    """
    timedate = time.strftime('%y%m%d_%H%M%S')
    cur_tb_path = parent_dir + '%s_%s_batchSize%s_layerSize%s_inputSize%s_learningRate%s_ep%s/' % (
                                                               timedate,
                                                               brnn_object.cell_type,
                                                               brnn_object.batch_size,
                                                               brnn_object.layer_size,
                                                               brnn_object.input_size,
                                                               brnn_object.learning_rate,
                                                               epoch_index)
    if os.path.isdir(cur_tb_path):
        shutil.rmtree(cur_tb_path)
    os.mkdir(cur_tb_path)
    return tf.summary.FileWriter(cur_tb_path, brnn_object.session.graph)


def plot_timeseries(raw, base_labels, y_hat, brnn_object):
    ts_plot = figure(title='Classified time series')
    ts_plot.grid.grid_line_alpha = 0.3
    ts_plot.xaxis.axis_label = 'nb events'
    ts_plot.yaxis.axis_label = 'current signal'
    y_range = raw.max() - raw.min()
    colors = ['#ffffff', '#fdcc8a', '#fc8d59', '#e34a33', '#b30000']
    # TODO: replace CategoricalColorMapper by LinearColorMapper
    col_mapper = CategoricalColorMapper(factors=list(range(brnn_object.num_classes)), palette=colors)
    source = ColumnDataSource(dict(
        raw=raw,
        event=list(range(len(y_hat))),
        cat=y_hat,
        cat_height=np.repeat(np.mean(raw), len(y_hat)),
        base_labels=base_labels
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
