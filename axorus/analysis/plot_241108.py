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

#%%

weighted_centres['d'] = np.sqrt(
    (weighted_centres['xc']-weighted_centres['laser_bin_x'])**2 +
    (weighted_centres['yc']-weighted_centres['laser_bin_y'])**2
) * 30


fig = utils.simple_fig(
    equal_width_height='y',
    width=1,
    height=1,
)

fig.add_scatter(
    x=weighted_centres['laser_bin_x'].values * 30,
    y=weighted_centres['d'].values,
    mode='markers',
    marker=dict(color='orange', size=6),
)

fig.update_xaxes(
    tickvals=np.arange(0, 300, 60),
    range=[0, 350],
    title_text='laser pos on x-axis [um]'
)

fig.update_yaxes(
    tickvals=np.arange(0, 400, 20),
    range=[0, 110],
    title_text=f'd (resp. - laser.) [um]'
)

sname = figure_dir / session_id / 'mea_scan' / f'dist_from_laser_centre'
utils.save_fig(fig, sname, display=True)


#%% heatmaps

heatmap_df = pd.DataFrame()
row_i = 0

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
        d = np.sqrt((lx - train_laser_x)**2 + (ly - train_laser_y)**2)

        x_rel = lx - train_laser_x
        y_rel = ly - train_laser_y

        if i not in cells_df.index.values:
            continue

        fr = cells_df.loc[i, (train_id, 'response_firing_rate')]
        is_sig = cells_df.loc[i, (train_id, 'is_significant')]

        if pd.isna(fr):
            continue

        heatmap_df.at[row_i, 'uid'] = i
        heatmap_df.at[row_i, 'tid'] = train_id
        heatmap_df.at[row_i, 'fr'] = fr
        heatmap_df.at[row_i, 'is_sig'] = is_sig
        heatmap_df.at[row_i, 'd'] = d
        heatmap_df.at[row_i, 'x_rel'] = x_rel
        heatmap_df.at[row_i, 'y_rel'] = y_rel

        row_i += 1

#%%
from scipy.ndimage import gaussian_filter


x = heatmap_df.x_rel.values
y = heatmap_df.y_rel.values
z = heatmap_df.fr.values


grid_binwidth = 15
axlen = 500
xmin = -axlen
xmax = axlen
ymin = -axlen
ymax = axlen
# binwidth = 50
n = int((xmax - xmin) / grid_binwidth)
xi = np.linspace(xmin, xmax, n+1)
yi = np.linspace(ymin, ymax, n+1)

# Create a grid for the heatmap
grid_size = 100
xi, yi = np.meshgrid(xi, yi)
xi = xi.astype(float)
yi = yi.astype(float)
zi = np.zeros_like(xi)
zi_count = np.zeros_like(xi)

# Populate the grid with the z values using nearest points
for i in range(len(x)):
    xi_idx = np.abs(xi[0] - x[i]).argmin()
    yi_idx = np.abs(yi[:, 0] - y[i]).argmin()
    if pd.notna(z[i]):
        zi[yi_idx, xi_idx] += z[i]
        zi_count[yi_idx, xi_idx] += 1

idx = zi_count > 0
zi[idx] = zi[idx] / zi_count[idx]

# Apply Gaussian filter to smooth the grid
zi = gaussian_filter(zi, sigma=3)


fig = utils.simple_fig(
    equal_width_height='y',
    width=0.5,
    height=1,
)

# Plot the heatmap
fig.add_heatmap(
    z=zi,
    x=xi[0],  # x-axis values
    y=yi[:, 0],  # y-axis values
    colorscale='Viridis',
    showscale=False,
)

for i in np.arange(-axlen, axlen, 250):
    fig.add_scatter(
        x=[i, i], y=[-axlen, axlen+1],
        mode='lines', line=dict(color='gold', dash='2px', width=0.5),
        showlegend=False,
    )
    fig.add_scatter(
        y=[i, i], x=[-axlen, axlen+1],
        mode='lines', line=dict(color='gold', dash='2px', width=0.5),
        showlegend=False,
    )

fig.update_xaxes(
    tickvals=np.arange(-axlen, axlen, 250),
    range=[-axlen, axlen]
)


fig.update_yaxes(
    tickvals=np.arange(-axlen, axlen, 250),
    range = [-axlen, axlen],
)


sname = figure_dir / session_id / 'mea_scan' / f'heatmap'
utils.save_fig(fig, sname, display=True)


#%%


fig = utils.simple_fig(
    equal_width_height='y',
    width=0.5,
    height=1,
)

clr = []
for i, r in heatmap_df.iterrows():
    if r.is_sig:
        clr.append('green')
    else:
        clr.append('black')


fig.add_scatter(
    x=heatmap_df.d.values,
    y=heatmap_df.fr.values,
    mode='markers',
    marker=dict(color=clr, size=2)
)

fig.update_xaxes(
    range=[0, 500],
    tickvals=np.arange(0, 500, 100),
    title_text=f'd. from laser [um]'
)

fig.update_yaxes(
    range=[0, 200],
    tickvals=np.arange(0, 200, 50),
    title_text=f'fr [Hz]'
)


sname = figure_dir / session_id / 'mea_scan' / f'fr_vs_dist'
utils.save_fig(fig, sname, display=True)


#%% Plot pa min max series
burst_df = data_io.burst_df.query('protocol == "pa_dc_min_max_series"')


# General setup of figure
electrodes = burst_df.electrode.unique()
n_rows = 3
n_cols = int(np.ceil(len(electrodes) / n_rows))

x_domains = {}
y_domains = {}
subplot_titles = {}
x_offset = 0.05
y_offset = 0.1
x_spacing = 0.1
y_spacing = 0.1

x_width = (1 - (n_cols-1) * x_spacing - 2 * x_offset) / n_cols
y_height = (1 - (n_rows - 1) * y_spacing - 2 * y_offset) / n_rows

electrode_i = 0
electrode_pos = {}

for row_i in range(n_rows):
    x_domains[row_i+1] = []
    y_domains[row_i+1] = []
    subplot_titles[row_i+1] = []

    for col_j in range(n_cols):
        x0 = x_offset + col_j * (x_spacing + x_width)
        x_domains[row_i+1].append([x0, x0+x_width])

        y1 = 1 - y_offset - row_i * (y_spacing + y_height)
        y_domains[row_i+1].append([y1 - y_height, y1])

        if electrode_i >= len(electrodes):
            subplot_titles[row_i+1].append(f'')
        else:
            subplot_titles[row_i+1].append(f'{electrodes[electrode_i]:.0f}')
            electrode_pos[electrodes[electrode_i]] = dict(row=row_i+1, col=col_j+1)
        electrode_i += 1

#%% Plot fraction of cells responding

for cluster_id in data_io.cluster_df.index.values:
    # Find distance from laser
    cx = data_io.cluster_df.loc[cluster_id].cluster_x
    cy = data_io.cluster_df.loc[cluster_id].cluster_y

    for electrode in electrode_pos.keys():
        r = electrode_pos[electrode]['row']
        c = electrode_pos[electrode]['col']

        lx = data_io.burst_df.query(f'electrode == {electrode}').iloc[0].laser_x
        ly = data_io.burst_df.query(f'electrode == {electrode}').iloc[0].laser_y

        d = np.sqrt((lx-cx)**2 + (ly-cy)**2)

        subplot_titles[r][c-1] = f'{electrode:.0f} - d = {d:.0f} um'


    fig = utils.make_figure(
        width=1, height=1.2,
        x_domains=x_domains,
        y_domains=y_domains,
        subplot_titles=subplot_titles,
    )

    fr_max_all = 0

    for electrode, df in burst_df.groupby('electrode'):
        pos = electrode_pos[electrode]

        n_pulses = []
        fr_baseline = []
        fr_response = []
        xticks = []

        for train_id, tdf in df.sort_values('duty_cycle').groupby('train_id'):

            rf = tdf.iloc[0].repetition_frequency / 1000
            bd = tdf.iloc[0].burst_duration

            n_pulses.append(rf * bd)
            xticks.append(f'{rf*bd:.0f}')

            # rf.append(f'{tdf.iloc[0].repetition_frequency/1000:.1f}')
            # xticks.append(f'{tdf.iloc[0].repetition_frequency/1000:.1f}')

            fr_baseline.append(cells_df.loc[cluster_id, (train_id, 'baseline_firing_rate')])
            fr_response.append(cells_df.loc[cluster_id, (train_id, 'response_firing_rate')])

        # convert lists to numpy arrays
        n_pulses = np.array(n_pulses)
        fr_baseline = np.array(fr_baseline)
        fr_response = np.array(fr_response)
        xticks = np.array(xticks)

        # sort by dc values
        sort_idx = np.argsort(n_pulses)
        fr_baseline = fr_baseline[sort_idx]
        fr_response = fr_response[sort_idx]
        n_pulses = n_pulses[sort_idx]
        xticks = xticks[sort_idx]

        # Plot baseline data
        fig.add_scatter(
            x=n_pulses, y=fr_baseline,
            mode='lines+markers',
            line=dict(color='grey', width=1),
            showlegend=True if pos['row'] == 1 and pos['col'] == 1 else False,
            name='baseline',
            **pos,
        )

        # Plot response data
        fig.add_scatter(
            x=n_pulses, y=fr_response,
            mode='lines+markers',
            line=dict(color='darkgreen', width=1),
            showlegend=True if pos['row'] == 1 and pos['col'] == 1 else False,
            name='response',
            **pos,
        )

        if np.all(pd.isna(fr_baseline)) and np.all(pd.isna(fr_response)):
            continue

        fr_max = np.nanmax([np.nanmax(fr_baseline), np.nanmax(fr_response)])
        fr_max_all = max(fr_max, fr_max_all)



    for col in range(n_cols):
        fig.update_xaxes(
            tickvals=xticks,
            title_text='n pulses in 10 ms',
            row=n_rows,
            col=col+1,
        )

    for electrode in electrode_pos.keys():
        row = electrode_pos[electrode]['row']
        col = electrode_pos[electrode]['col']

        if fr_max_all < 10:
            tickvals = np.arange(0, 12, 1)
            r =[0, 10]
        elif fr_max_all < 20:
            tickvals = np.arange(0, 22, 5)
            r = [0, 20]
        elif fr_max_all < 60:
            tickvals = np.arange(0, 62, 20)
            r = [0, 60]
        elif fr_max_all < 100:
            tickvals = np.arange(0, 101, 50)
            r = [0, 101]
        elif fr_max_all < 200:
            tickvals = np.arange(0, 201, 50)
            r = [0, 201]
        else:
            tickvals = np.arange(0, 501, 100)
            r =[0, 501]

        fig.update_yaxes(
            tickvals=tickvals,
            range=r,
            row=row, col=col,
        )


    sname = figure_dir / session_id / 'pa_dc_min_max' /  'curve_per_cell' /  cluster_id
    utils.save_fig(fig, sname, display=False)

#%% Plot fraction of significantly responding cells as function of nr pulses

for electrode in electrode_pos.keys():
    r = electrode_pos[electrode]['row']
    c = electrode_pos[electrode]['col']
    subplot_titles[r][c-1] = f'{electrode:.0f}'



fig = utils.make_figure(
    width=1, height=1.2,
    x_domains=x_domains,
    y_domains=y_domains,
    subplot_titles=subplot_titles,
)

for electrode, df in burst_df.groupby('electrode'):
    pos = electrode_pos[electrode]

    n_pulses = []
    n_responding = []
    xticks = []

    for train_id, tdf in df.sort_values('duty_cycle').groupby('train_id'):
        rf = tdf.iloc[0].repetition_frequency / 1000
        bd = tdf.iloc[0].burst_duration

        n_pulses.append(rf * bd)
        xticks.append(f'{rf * bd:.0f}')

        n = 0
        for cid in data_io.cluster_df.index.values:
            if cells_df.loc[cid, (train_id, 'is_significant')]:
                n += 1

        n_responding.append(n)

    # convert lists to numpy arrays
    n_pulses = np.array(n_pulses)
    n_responding = np.array(n_responding)
    xticks = np.array(xticks)

    # sort by dc values
    sort_idx = np.argsort(n_pulses)
    n_responding = n_responding[sort_idx]
    n_pulses = n_pulses[sort_idx]
    xticks = xticks[sort_idx]

    # Plot baseline data
    fig.add_scatter(
        x=n_pulses, y=n_responding,
        mode='lines+markers',
        line=dict(color='black', width=1),
        showlegend=False,
        **pos,
    )

    fig.update_yaxes(
        tickvals=np.arange(0, 80, 20),
        range=[0, 80],
        title_text='# cells resp.' if pos['col'] == 1 else '',
    )

for col in range(n_cols):
    fig.update_xaxes(
        tickvals=xticks,
        title_text='n pulses in 10 ms',
        row=n_rows,
        col=col+1,
    )

sname = figure_dir / session_id / 'pa_dc_min_max' / 'count_responding_per_dc'
utils.save_fig(fig, sname, display=True)

#%% plot waveforms

for cluster_id, cinfo in data_io.cluster_df.iterrows():
    fig = utils.make_figure(
        width=1,
        height=1,
        x_domains={
            1: [[0.1, 0.4], [0.6, 0.9]],
        },
        y_domains={
            1: [[0.1, 0.9], [0.1, 0.9]]
        },
        subplot_titles={
            1: data_io.recording_ids,
        }
    )

    for ri, rid in enumerate(data_io.recording_ids):

        wv = data_io.waveforms[rid][cluster_id]

        if wv.size == 0:
            continue

        pos = dict(row=1, col=ri+1)

        x_plot = []
        y_plot = []
        n_pts = wv.shape[1]
        for wave_i in range(wv.shape[0]):
            x_plot.append(np.arange(0, n_pts, 1) / 20)
            x_plot.append([None])

            y = wv[wave_i, :] - np.mean(wv[wave_i, :])
            y_plot.append(y)
            y_plot.append([None])

        x_plot = np.hstack(x_plot)
        y_plot = np.hstack(y_plot)

        fig.add_scatter(
            x=x_plot, y=y_plot,
            mode='lines', line=dict(color='black', width=0.1),
            showlegend=False,
            **pos,
        )

        fig.update_xaxes(
            tickvals=np.arange(0, 30, 5),
            title_text='time [ms]'
        )
        fig.update_yaxes(
            range=[-250, 200],
            tickvals=np.arange(-400, 400, 100),
            **pos)

    sname = figure_dir / session_id / 'waveforms' / f'{cluster_id}'
    utils.save_fig(fig, sname, display=False)
    # break




