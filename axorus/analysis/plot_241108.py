import pandas as pd
from pathlib import Path

from matplotlib.pyplot import subplot

from axorus.data_io import DataIO
import utils
import numpy as np
from axorus.preprocessing.params import data_sample_rate, data_type, data_nb_channels

session_id = '241108_A'
data_dir = Path(r'D:\Axorus\ex_vivo_series_3\dataset')
figure_dir = Path(r'C:\Axorus\figures')
data_io = DataIO(data_dir)
loadname = data_dir / f'{session_id}_cells.csv'

data_io.load_session(session_id)

cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)


#%% Visualize the MEA scan

trials = data_io.burst_df.query('protocol == "pa_mea_scan"')

laser_x = trials.laser_x.unique()
laser_x = np.sort(laser_x)
laser_x = laser_x
n_cols = len(laser_x)

laser_y = trials.laser_y.unique()
laser_y = np.sort(laser_y)[::-1]
n_rows = len(laser_y)

xmin = data_io.cluster_df.cluster_x.min()
xmax = data_io.cluster_df.cluster_x.max()
ymin = data_io.cluster_df.cluster_y.min()
ymax = data_io.cluster_df.cluster_y.max()
xbins = np.arange(xmin, xmax + 35, 30)
ybins = np.arange(ymin, ymax + 35, 30)



xmap = np.zeros((xbins.size, ybins.size))
ymap = np.zeros((xbins.size, ybins.size))
for i in range(xmap.shape[1]):
    xmap[:, i] = xbins
for i in range(ymap.shape[0]):
    ymap[i, :] = ybins



def weighted_center(heatmap):
    n_x, n_y = heatmap.shape
    x_indices, y_indices = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing='ij')
    total_weight = np.sum(heatmap)
    x_center = np.sum(x_indices * heatmap) / total_weight
    y_center = np.sum(y_indices * heatmap) / total_weight
    return x_center, y_center



x_domains = {}
y_domains = {}
x_offset = 0.05
x_spacing = 0.01
x_width = (1 - ((n_cols-1)*x_spacing) - 2 * x_offset) / n_cols
y_offset = 0.05
y_spacing = 0.01
y_height = (1 - ((n_rows - 1) * y_spacing) - 2 * y_offset) / n_rows
x0 = laser_x[0]
y0 = laser_y[0]

for row_i in range(n_rows):
    y1 = 1 - y_offset - row_i * (y_spacing + y_height)
    y_domains[row_i+1] = [[y1-y_height, y1] for _ in range(n_cols)]
    x_domains[row_i+1] = []
    for col_i in range(n_cols):
        x0 = x_offset + col_i * (x_spacing + x_width)
        x_domains[row_i+1].append([x0, x0+x_width])

fig = utils.make_figure(
    width=2,
    height=1.8,
    x_domains=x_domains,
    y_domains=y_domains,
    equal_width_height='x',
)

weighted_centres = pd.DataFrame()
df_i = 0

for train_id in trials.train_id.unique():

    train_info = trials.query(f'train_id == "{train_id}"')
    train_laser_x = train_info.iloc[0].laser_x
    train_laser_y = train_info.iloc[0].laser_y

    x_idx = int(np.argwhere(laser_x == train_laser_x)[0][0])
    y_idx = int(np.argwhere(laser_y == train_laser_y)[0][0])
    row = x_idx + 1
    col = y_idx + 1
    pos = dict(row=col, col=row)

    fr_sum = np.zeros((xbins.size, ybins.size))
    fr_count = np.zeros((xbins.size, ybins.size))
    fr_map = np.zeros((xbins.size, ybins.size))

    for i, r in data_io.cluster_df.iterrows():
        lx = r.cluster_x
        ly = r.cluster_y

        dx = np.abs(xbins - lx)
        dy = np.abs(ybins - ly)
        xi = int(np.argmin(dx))
        yi = int(np.argmin(dy))

        if i not in cells_df.index.values:
            continue
        fr = cells_df.loc[i, (train_id, 'response_firing_rate')]
        if pd.isna(fr):
            continue

        fr_sum[xi, yi] += fr
        fr_count[xi, yi] += 1

    fr_map[fr_count > 0] = fr_sum[fr_count > 0] / fr_count[fr_count >0]

    # Plot the heatmap
    fig.add_heatmap(
        z=fr_map.T,
        # x=ybins,  # x-axis values
        # y=xbins,  # y-axis values
        colorscale='Viridis',
        showscale=False,
        **pos,
    )

    # Plot weighted centre
    xc, yc = weighted_center(fr_map)
    dx = np.abs(xbins - train_laser_x)
    dy = np.abs(ybins - train_laser_y)
    xi = int(np.argmin(dx))
    yi = int(np.argmin(dy))

    weighted_centres.at[df_i, 'xc'] = xc
    weighted_centres.at[df_i, 'yc'] = yc
    weighted_centres.at[df_i, 'laser_bin_x'] = xi
    weighted_centres.at[df_i, 'laser_bin_y'] = yi

    df_i += 1

    fig.add_scatter(
        y=[yc], x=[xc],
        mode='markers', marker=dict(color='orange', size=6),
        showlegend=False,
        **pos,
    )


    fig.add_scatter(
        y=[yi], x=[xi],
        mode='markers', marker=dict(color='red', size=4),
        showlegend=False,
        **pos,
    )


    for yy, yclr in zip([120, 180, 240], ['gold', 'gold', 'gold']):
        # dx = np.abs(xbins - 120)
        dy = np.abs(ybins - yy)
        # xi = int(np.argmin(dx))
        yi = int(np.argmin(dy))

        fig.add_scatter(
            y=[yi, yi], x=[0, fr_map.shape[0]],
            mode='lines', line=dict(color=yclr, width=0.5, dash='2px'),
            showlegend=False,
            **pos,
        )


    fig.update_xaxes(
        range=[-0.5, fr_map.shape[0] - 0.5],
        **pos,
    )
    fig.update_yaxes(
        range=[-0.5, fr_map.shape[1] - 0.5],
        **pos,
    )


sname = figure_dir / session_id / 'mea_scan' / f'mea_scan'
utils.save_fig(fig, sname, display=True)


#%%

fig = utils.simple_fig(
    equal_width_height='y',
    width=1,
    height=1,
)

for y, ydf in weighted_centres.groupby('laser_bin_y'):
    df2 = ydf.sort_values('laser_bin_x')
    x = df2.iloc[0].xc
    yy = df2.iloc[0].yc

    d = np.sqrt((df2.xc - x)**2 + (df2.yc - yy)**2)
    print(d.values)

    fig.add_scatter(x=df2.laser_bin_x * 30, y=d*30, showlegend=False,)

fig.update_xaxes(
    title_text=f'shift x-axes [um]',
    tickvals=np.arange(0, 350, 50),
    range=[0, 300]
)

fig.update_yaxes(
    tickvals=np.arange(0, 350, 50),
    title_text=f'shift in centre of response [um]',
    range=[0, 300]
)

sname = figure_dir / session_id / 'mea_scan' / f'wc_distances'
utils.save_fig(fig, sname, display=True)







