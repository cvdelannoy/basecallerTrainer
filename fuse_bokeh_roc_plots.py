import os
import re

from bokeh.models import ColumnDataSource, CategoricalColorMapper, LabelSet, Range1d
from bokeh.plotting import figure
from bokeh.io import save, output_file
from math import pi
from helper_functions import parse_output_path
from scipy.spatial import ConvexHull

categorical_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']

def plot_combined_roc_curve(roc_list, title):
    tpr, fpr, dif = zip(*roc_list)

    # # Calculate convex hull points
    # ch_tpr = [0] + list(tpr) + [1]
    # ch_fpr = [0] + list(fpr) + [1]
    # vertices_idx = ConvexHull(list(zip(ch_tpr, ch_fpr))).vertices
    # ch_fpr = [ch_fpr[i] for i in vertices_idx]
    # ch_tpr = [ch_tpr[i] for i in vertices_idx]
    # ch_points = [(ch_fpr[i], ch_tpr[i]) for i in vertices_idx if ch_fpr[i] ]
    # plot( )

    roc_plot = figure(title=title)
    roc_plot.grid.grid_line_alpha = 0.3
    roc_plot.xaxis.axis_label = 'FPR'
    roc_plot.yaxis.axis_label = 'TPR'

    dif_unique = list(set(dif))
    colors = categorical_colors[:len(dif_unique)]
    col_mapper = CategoricalColorMapper(factors= dif_unique, palette=colors)

    source = ColumnDataSource(dict(
        TPR=tpr,
        FPR=fpr,
        dif=dif
    ))
    # roc_plot.line(x=[ch_fpr[i] for i in vertices_idx],
    #               y=[ch_tpr[i] for i in vertices_idx], color='grey')
    roc_plot.scatter(x='FPR', y='TPR',
                     source=source,
                     color={'field': 'dif',
                            'transform': col_mapper},
                     legend='dif')
    roc_plot.ray(x=0, y=0, length=1.42, angle=0.25*pi, color='grey')
    roc_plot.x_range = Range1d(0, 1)
    roc_plot.y_range = Range1d(0, 1)
    roc_plot.plot_width = 500
    roc_plot.plot_height = 500
    roc_plot.legend.location = 'bottom_right'
    return roc_plot


plots_path = '/mnt/nexenta/lanno001/nobackup/additional_graphs/'
roc_plots_path = '/mnt/nexenta/lanno001/nobackup/roc_combined/'
roc_plots_path = parse_output_path(roc_plots_path)


FPR_all = []
TPR_all = []
epoch_all = []
cell_type_all = []
optimizer_all = []
learning_rate_all = []
layer_size_all = []
num_layers_all = []
batch_size_all = []

folders = os.listdir(plots_path)
for folder in folders:
    folder_path = plots_path + folder
    sub_folder = os.listdir(folder_path)[0]
    sub_folder_path = folder_path + '/' + sub_folder
    run_name = os.path.basename(sub_folder)
    try:
        with open(sub_folder_path+"/roc_end.html", 'r') as f:
            roc_content = f.read()
    except FileNotFoundError:
        continue
    TPR = re.search('(?<="TPR":\[)[0-9.,]+', roc_content).group()
    TPR = list(map(float,TPR.split(',')))
    TPR_all.extend(TPR)
    FPR = re.search('(?<="FPR":\[)[0-9.,]+', roc_content).group()
    FPR = list(map(float, FPR.split(',')))
    FPR_all.extend(FPR)
    num_points = len(FPR)
    epoch = re.search('(?<="epoch":\[)[0-9,]+', roc_content).group()
    epoch = list(map(float, epoch.split(',')))
    epoch_all.extend(epoch)
    cell_type = re.search('(LSTM)|(GRU)', run_name).group()
    cell_type_all.extend([cell_type] * num_points)
    optimizer = re.search('(adadelta)|(adam)', run_name).group()
    optimizer_all.extend([optimizer] * num_points)
    batch_size = re.search('(?<=bs)[0-9]+', run_name).group()
    batch_size_all.extend([batch_size] * num_points)
    learning_rate = re.search('(?<=lr)[0-9.]+', run_name).group()
    learning_rate_all.extend([learning_rate] * num_points)
    layer_size = re.search('(?<=ls)[0-9]+', run_name).group()
    layer_size_all.extend([layer_size] * num_points)
    num_layers = re.search('(?<=numl)[0-9]', run_name).group()
    num_layers_all.extend([num_layers] * num_points)

dif_dict = dict(epoch=epoch_all,
                cell_type=cell_type_all,
                optimizer=optimizer_all,
                batch_size= batch_size_all,
                learning_rate=learning_rate_all,
                layer_size=layer_size_all,
                number_of_layers=num_layers_all)

for dif in dif_dict:
    roc_epoch = plot_combined_roc_curve(zip(TPR_all, FPR_all, dif_dict[dif]), dif.replace('_', ' '))
    output_file(roc_plots_path+dif+'.html')
    save(roc_epoch)
