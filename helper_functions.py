import tensorflow as tf
import numpy as np
import os
import shutil
import time
import warnings
import itertools
import re
import h5py
from math import nan

from bokeh.models import ColumnDataSource, LinearColorMapper, LabelSet, Range1d
from bokeh.plotting import figure
# from bokeh.io import show

from math import pi


categorical_colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072']
continuous_colors = ['#ffffff', '#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84',
                     '#fc8d59', '#ef6548', '#d7301f', '#990000']


def parse_input_path(location):
    """
    Take path, list of files or single file. Add '/' if path. Return list of files with path name concatenated. 
    """
    if not isinstance(location, list):
        location = [location]

    all_files = []
    for loc in location:
        if os.path.isdir(loc):
            if loc[-1] != '/':
                loc += '/'
            file_names = os.listdir(loc)
            files = [loc + f for f in file_names]
            all_files.extend(files)
        elif os.path.exists(loc):
            all_files.extend(loc)
        else:
            warnings.warn('Given location %s does not exist, skipping' % loc, RuntimeWarning)

    if not len(all_files):
        ValueError('Input file location(s) did not exist or did not contain any files.')
    return all_files


def parse_output_path(location):
    """
    Take given path name. Add '/' if path. Check if exists, if not, make dir and subdirs. 
    """
    if location[-1] != '/':
        location += '/'
    if not os.path.isdir(location):
        os.makedirs(location)
    return location

def set_logfolder(brnn_object, parent_dir, epoch_index):
    """
    Create a folder to store tensorflow metrics for tensorboard and set it up for a specific session.
    Returns a filewriter object, which can be used to write info to tensorboard.
    """
    timedate = time.strftime('%y%m%d_%H%M%S')
    cur_tb_path = parent_dir + '%s_%s_batchSize%s_learningRate%s_layerSize%s_%s_numLayers%s_ep%s/' % (
                                                               timedate,
                                                               brnn_object.cell_type,
                                                               brnn_object.batch_size,
                                                               brnn_object.learning_rate,
                                                               brnn_object.layer_size,
                                                               brnn_object.name_optimizer,
                                                               brnn_object.num_layers,
                                                               epoch_index)
    if os.path.isdir(cur_tb_path):
        shutil.rmtree(cur_tb_path)
    os.makedirs(cur_tb_path)
    return tf.summary.FileWriter(cur_tb_path, brnn_object.session.graph)

def plot_timeseries(raw, base_labels, y_hat, brnn_object, categorical=False):
    ts_plot = figure(title='Classified time series')
    ts_plot.grid.grid_line_alpha = 0.3
    ts_plot.xaxis.axis_label = 'nb events'
    ts_plot.yaxis.axis_label = 'current signal'
    y_range = raw.max() - raw.min()
    if categorical:
        colors = categorical_colors
    else:
        colors = continuous_colors
    col_mapper = LinearColorMapper(palette=colors, low=1, high=brnn_object.num_classes)
    # col_mapper = CategoricalColorMapper(factors=list(range(brnn_object.num_classes)), palette=colors)
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
    if categorical:
        base_labels_labelset = LabelSet(x='event', y='cat_height',
                                        y_offset=-y_range,
                                        text='base_labels', text_baseline='middle',
                                        source=source)
    else:
        base_labels_labelset = LabelSet(x='event', y='cat_height',
                                        y_offset=-y_range, angle=-0.5 * pi,
                                        text='base_labels', text_baseline='middle',
                                        source=source)
    ts_plot.add_layout(base_labels_labelset)
    ts_plot.scatter(x='event', y='raw', source=source)
    ts_plot.plot_width = 1000
    ts_plot.plot_height = 500
    ts_plot.x_range = Range1d(0, 500)
    return ts_plot


def plot_roc_curve(roc_list):
    tpr, tnr, epoch = zip(*roc_list)
    roc_plot = figure(title='ROC')
    roc_plot.grid.grid_line_alpha = 0.3
    roc_plot.xaxis.axis_label = 'FPR'
    roc_plot.yaxis.axis_label = 'TPR'

    col_mapper = LinearColorMapper(palette=categorical_colors, low=1, high=max(epoch))
    source = ColumnDataSource(dict(
        TPR=tpr,
        FPR=[1-cur_tnr for cur_tnr in tnr],
        epoch=epoch
    ))
    roc_plot.scatter(x='FPR', y='TPR',
                     color={'field': 'epoch',
                            'transform': col_mapper},
                     source=source)
    roc_plot.ray(x=0, y=0, length=1.42, angle=0.25*pi, color='grey')
    roc_plot.x_range = Range1d(0, 1)
    roc_plot.y_range = Range1d(0, 1)
    roc_plot.plot_width = 500
    roc_plot.plot_height = 500
    return roc_plot


def retrieve_read_properties(raw_read_dir, read_name):
    read_name_grep = re.search('(?<=/)[^/]+_strand', read_name).group()
    # Reconstruct full read name + path
    fast5_name = raw_read_dir + read_name_grep + '.fast5'
    try:
        hdf = h5py.File(fast5_name, 'r')
    except OSError:
        warnings.warn('Read %s not found in raw data, skipping reat property retrieval.' % fast5_name, RuntimeWarning)
        return [nan for _ in range(5)]


    # Get metrics
    qscore = hdf['Analyses/Basecall_1D_000/Summary/basecall_1d_template'].attrs['mean_qscore']
    alignment = hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment']
    alignment_metrics = [alignment.attrs[n] for n in ('num_deletions',
                                                      'num_insertions',
                                                      'num_mismatches',
                                                      'num_matches')]
    hdf.close()
    return [qscore] + alignment_metrics


def normalize_raw_signal(raw, norm_method):
    """
    Normalize the raw DAC values
     
    """
    # Median normalization, as done by nanoraw (see nanoraw_helper.py)
    if norm_method == 'median':
        shift = np.median(raw)
        scale = np.median(np.abs(raw - shift))
    else:
        raise ValueError('norm_method not recognized')
    return (raw - shift) / scale


# def simulate_on_the_fly(tr_name):
#     if params['simulate_on_the_fly']:
#         readnb = int(re.search("(?<=read)\d+(?!(.*/))", tr).group())
#         hdf = h5py.File(tr, 'r')
#         try:
#             tr_cur = trainingRead.TrainingRead(hdf, readnb, 'median', use_nanoraw=False)
#         except KeyError:
#             hdf.close()
#             continue
#         encoded = tr_cur.classify_events('hp_5class')
#         if tr_cur.events is None:
#             continue
#         unknown_index = tr_cur.events != 'NNNNN'  # Remove data for raw data without a k-mer
#         raw = tr_cur.raw[unknown_index]
#         onehot = encoded[unknown_index]
#         frac_min_class = np.sum(onehot == 5) / encoded.size
#         if frac_min_class < min_content_percentage or raw.size < params['read_length']:
#             continue


def apply_post_processing(y_hat):
    """
    Check for transitions that are not allowed and correct them 
    """

    # condense
    y_cd = [(cls, len(list(n))) for cls, n in itertools.groupby(y_hat)]

    y_cor = []
    for ci in (range(1,len(y_cd)-1)):
        prev = y_cd[ci-1]
        cur = y_cd[ci]
        next = y_cd[ci+1]

        # For now, only correct 5
        if cur[0] == 5:
            if prev[0] != 4 and next[0] == 4:
                if cur[1] <= 3 and cur[1] < prev[1]:
                    cur = (4, cur[1])
                elif prev[1] <= 3 and cur[1] > prev[1]:
                    prev = (5, prev[1])
            elif prev[0] == 4 and next[0] != 4:
                if cur[1] <= 3 and cur[1] < next[1]:
                    cur = (4, cur[1])
                elif next[1] <= 3 and cur[1] > next[1]:
                    next = (5, next[1])
            elif prev[0] != 4 and next[0] != 4:
                if cur[1] <= 3 and prev[0] == next[0]:
                    cur = (prev[0], cur[1])
