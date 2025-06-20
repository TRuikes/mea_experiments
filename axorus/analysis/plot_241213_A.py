# %% General setup
import pandas as pd
from pathlib import Path
from axorus.preprocessing.project_colors import ProjectColors
from axorus.data_io import DataIO
import utils
import numpy as np
import axorus.analysis.figure_library as fl

# Load data
session_id = '241213_A'
# data_dir = Path(r'E:\Axorus\ex_vivo_series_3\dataset')
data_dir = Path(r'C:\axorus\tmp')
figure_dir = Path(r'C:\Axorus\figures')
data_io = DataIO(data_dir)
loadname = data_dir / f'{session_id}_cells.csv'
data_io.load_session(session_id, load_pickle=True)
cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
clrs = ProjectColors()

INCLUDE_RANGE = 50  # include cells at max distance = 50 um

#%%

names_to_get = (
    ('burst_count', 'burst_count'),
    ('burst_duration', 'burst_duration'),
    ('e_pulse', 'e_pulse'),
    ('irradiance', 'irradiance_3x_fiber_diameter'),
    ('repetition_frequency', 'repetition_frequency'),
    ('protocol', 'protocol')
)

df_out = pd.DataFrame()
for tid, tdf in data_io.burst_df.groupby('train_id'):
    # print(tid)

    for sname, dname in names_to_get:
        df_out.at[tid, sname] = tdf.iloc[0][dname]


savename = data_dir / f'{session_id}_trials.csv'
df_out.to_csv(savename)

#%% 1 WAVEFORMS

savedir = figure_dir / session_id / 'waveforms'
d2 = DataIO(data_dir)
d2.load_session(session_id, load_pickle=False, load_waveforms=True)

print(d2.recording_ids)
#%%

fl.plot_session_waveforms(d2, savedir, rec_nrs=[1, 3])


#%% 2 PA MIN MAX SERIES

burst_df = data_io.burst_df.query('protocol == "pa_dc_min_max_series"')

print('Np\t| Prf\t\t| Pe\t\t| Irradiance\t\t| burst duration')
for dc, dc_df in burst_df.groupby('duty_cycle'):
    e_pulse = dc_df.iloc[0].e_pulse
    frep = dc_df.iloc[0].repetition_frequency
    irr_exact = dc_df.iloc[0].irradiance_exact_fiber_diameter
    irr_min_3x = dc_df.iloc[0].irradiance_3x_fiber_diameter
    bd = dc_df.iloc[0].burst_duration
    npulses = (frep / 1000) * bd

    print(f'{npulses:.0f}\t| {frep:4.0f} Hz \t| {e_pulse:.2f} uJ\t| {irr_exact:.2f} - {irr_min_3x:.2f} W/mm2\t| {bd:.0f} ms')


#%% 2.1 heatmaps

# Plot heatmaps for 1 Prf
Prf = data_io.burst_df.repetition_frequency.max()
burst_df = data_io.burst_df.query(f'repetition_frequency == {Prf}')
train_ids = burst_df.train_id.unique()

xbins, ybins, heatmap = fl.get_heatmap_relative_to_laser(
    data_io, cells_df, train_ids,
    binsize=30,
    significant_only=False
)


fig = utils.simple_fig(
    equal_width_height='y',
    width=0.5,
    height=1,
)

# Plot the heatmap
fig.add_heatmap(
    z=heatmap,
    x=xbins,  # x-axis values
    y=ybins,  # y-axis values
    colorscale='Viridis',
    showscale=False,
)
axtickstep = 100



sname = figure_dir / session_id / 'pa_dc_min_max' / f'test'
utils.save_fig(fig, sname, display=True)


#%% 2.1 single cell fr vs pulse rate
# General setup of figure


#%% 2.2 fraction of cells responding, firing rate and latency, per electrode

burst_df = data_io.burst_df.query('protocol == "pa_dc_min_max_series"')


print('Generating figure...')
fig = utils.make_figure(
    width=0.6, height=1.4,
    x_domains={
        1: [[0.1, 0.9]],
        2: [[0.1, 0.9]],
        3: [[0.1, 0.9]],
        4: [[0.1, 0.9]],

    },
    y_domains={
        1: [[0.7, 0.85]],
        2: [[0.5, 0.65]],
        3: [[0.3, 0.45]],
        4: [[0.1, 0.25]],

    },
)

print('loading counts per stim site')
# Plot franction of responding cells per electrode
for electrode, df in burst_df.groupby('electrode'):
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
            if (cells_df.loc[cid, (train_id, 'laser_distance')] < INCLUDE_RANGE and
                    cells_df.loc[cid, (train_id, 'is_significant')]):
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
        row=1, col=1,
    )

fig.update_yaxes(
    tickvals=np.arange(0, 20, 2),
    # range=[0, 80],
    title_text='# cells resp.',
    row=1, col=1,
)

fig.update_xaxes(
    tickvals=xticks,
    title_text='',
    row=1, col=1,
)


print('loading fractions')
# Plot franction of responding cells per electrode
for electrode, df in burst_df.groupby('electrode'):
    n_pulses = []
    n_responding = []
    xticks = []

    for train_id, tdf in df.sort_values('duty_cycle').groupby('train_id'):
        rf = tdf.iloc[0].repetition_frequency / 1000
        bd = tdf.iloc[0].burst_duration

        n_pulses.append(rf * bd)
        xticks.append(f'{rf * bd:.0f}')

        n = cells_df[train_id].query(f'laser_distance <= {INCLUDE_RANGE}')['is_significant'].dropna().sum()
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

    n_responding = n_responding / np.max(n_responding)

    # Plot baseline data
    fig.add_scatter(
        x=n_pulses, y=n_responding,
        mode='lines+markers',
        line=dict(color='black', width=1),
        showlegend=False,
        row=2, col=1,
    )

fig.update_yaxes(
    tickvals=np.arange(0, 2, 0.2),
    range=[0, 1.1],
    title_text='frac. c. resp.',
    row=2, col=1,
)

fig.update_xaxes(
    tickvals=xticks,
    title_text='',
    row=2, col=1,
)

# plot firing rates
print('loading firing rates')
for electrode, df in burst_df.groupby('electrode'):

    n_pulses = []
    firing_rates = []
    xticks = []

    for train_id, tdf in df.sort_values('duty_cycle').groupby('train_id'):
        rf = tdf.iloc[0].repetition_frequency / 1000
        bd = tdf.iloc[0].burst_duration
        dc = tdf.iloc[0].duty_cycle

        n_pulses.append(rf * bd)
        xticks.append(f'{rf * bd:.0f} ({dc:.0f})')
        fr_mean = cells_df[train_id].query(f'is_significant == True and laser_distance <= {INCLUDE_RANGE}').response_firing_rate.mean()
        firing_rates.append(fr_mean)

    # convert lists to numpy arrays
    n_pulses = np.array(n_pulses)
    firing_rates = np.array(firing_rates)
    xticks = np.array(xticks)

    # sort by dc values
    sort_idx = np.argsort(n_pulses)
    firing_rates = firing_rates[sort_idx]
    n_pulses = n_pulses[sort_idx]
    xticks = xticks[sort_idx]

    # Plot baseline data
    fig.add_scatter(
        x=n_pulses, y=firing_rates,
        mode='lines+markers',
        line=dict(color='black', width=1),
        showlegend=False,
        row=3, col=1,
    )

fig.update_yaxes(
    tickvals=np.arange(0, 300, 50),
    ticklen=2,
    tickwidth=0.5,
    # range=[0, 80],
    title_text='fr [Hz]',
    row=3, col=1,
)

fig.update_xaxes(
    tickvals=n_pulses,
    ticktext=xticks,
    title_text='',
    col=1, row=3,
)

# plot latencies
print('loading latencies')
for electrode, df in burst_df.groupby('electrode'):

    n_pulses = []
    latencies = []
    xticks = []

    for train_id, tdf in df.sort_values('duty_cycle').groupby('train_id'):
        rf = tdf.iloc[0].repetition_frequency / 1000
        bd = tdf.iloc[0].burst_duration
        dc = tdf.iloc[0].duty_cycle

        n_pulses.append(rf * bd)
        xticks.append(f'{rf * bd:.0f} ({dc:.0f})')

        lat_trail = cells_df[train_id].query(f'is_significant == True and laser_distance <= '
                                             f'{INCLUDE_RANGE}').response_latency.mean()

        latencies.append(lat_trail)

    # convert lists to numpy arrays
    n_pulses = np.array(n_pulses)
    latencies = np.array(latencies)
    xticks = np.array(xticks)

    # sort by dc values
    sort_idx = np.argsort(n_pulses)
    latencies = latencies[sort_idx]
    n_pulses = n_pulses[sort_idx]
    xticks = xticks[sort_idx]

    # Plot baseline data
    fig.add_scatter(
        x=n_pulses, y=latencies,
        mode='lines+markers',
        line=dict(color='black', width=1),
        showlegend=False,
        row=4, col=1,
    )

fig.update_yaxes(
    tickvals=np.arange(0, 300, 50),
    ticklen=2,
    tickwidth=0.5,
    range=[50, 120],
    title_text='lat [ms]',
    row=4, col=1,
)

fig.update_xaxes(
    tickvals=n_pulses,
    ticktext=xticks,
    title_text='n pulses in 10 ms (dc)',
    col=1, row=4,
)

sname = figure_dir / session_id / 'pa_dc_min_max' / 'fraction_fr_lat_per_electrode'
utils.save_fig(fig, sname, display=True)

#%%
# Rasters for single cells at

tids = data_io.burst_df.query('protocol == "pa_dc_min_max_series" and duty_cycle == 5').train_id.unique()

c_per_trial = {}

for t in tids:
    for k, v in cells_df[t].iterrows():
        c_per_trial[t] = []
        if v.is_significant:
            c_per_trial[t].append(k)



train_id = 'tid_241108_A_013'
uids = cells_df[train_id].query('is_significant == True').index.tolist()

for c in uids:
    train_period = data_io.burst_df.query(f'train_id == "{train_id}"').iloc[0].train_period

    cluster_data = utils.load_obj(data_dir / 'bootstrapped' / f'bootstrap_{c}.pkl')

    stimes = cluster_data[t]['spike_times']
    bins = cluster_data[t]['bins']
    x_plot, y_plot = [], []
    row_i = 0

    for burst_i, sp in enumerate(stimes):
        # sp_to_plot = sp[(sp > 0) & (sp < train_period)]
        sp_to_plot = sp
        x_plot.append(np.vstack([sp_to_plot, sp_to_plot, np.full(sp_to_plot.size, np.nan)]).T.flatten())
        y_plot.append(np.vstack([np.ones(sp_to_plot.size) * burst_i,
                                 np.ones(sp_to_plot.size) * burst_i + 1, np.full(
                sp_to_plot.size, np.nan)]).T.flatten())

        row_i += 1

    x_plot = np.hstack(x_plot)
    y_plot = np.hstack(y_plot)

    fig = utils.simple_fig(
        width=0.5, height=0.5
    )

    fig.add_scatter(
        x=x_plot, y=y_plot,
        mode='lines', line=dict(color='black', width=0.5),
        showlegend=False,
    )

    fig.add_scatter(
        x=[0, 0], y=[0, row_i],
        mode='lines', line=dict(color='red'),
        showlegend=False,
    )

    fig.update_xaxes(
        range=[-200, 400],
        tickvals=np.arange(-600, 600, 100)
    )

    fig.update_yaxes(
        range=[0, row_i]
    )

    sname = figure_dir / session_id / 'pa_dc_min_max' / 'single_cell_rasters' / train_id / c
    utils.save_fig(fig, sname, display=False,)


#%% 2.4 firing rate vs distance from laser; per dc

distance_df = pd.DataFrame()
trials = data_io.burst_df.query('protocol == "pa_dc_min_max_series"')

row_i = 0

for train_id in trials.train_id.unique():

    train_info = trials.query(f'train_id == "{train_id}"')
    train_laser_x = train_info.iloc[0].laser_x
    train_laser_y = train_info.iloc[0].laser_y

    rf = train_info.iloc[0].repetition_frequency / 1000
    bd = train_info.iloc[0].burst_duration

    n_pulses = rf * bd

    for i, r in data_io.cluster_df.iterrows():
        lx = r.cluster_x
        ly = r.cluster_y
        d = np.sqrt((lx - train_laser_x)**2 + (ly - train_laser_y)**2)

        if i not in cells_df.index.values:
            continue

        fr = cells_df.loc[i, (train_id, 'response_firing_rate')]
        fr_base = cells_df.loc[i, (train_id, 'baseline_firing_rate')]
        is_sig = cells_df.loc[i, (train_id, 'is_significant')]

        if pd.isna(fr):
            continue

        distance_df.at[row_i, 'uid'] = i
        distance_df.at[row_i, 'tid'] = train_id
        distance_df.at[row_i, 'fr'] = fr
        distance_df.at[row_i, 'is_sig'] = is_sig
        distance_df.at[row_i, 'd'] = d
        distance_df.at[row_i, 'dc'] = train_info.iloc[0].duty_cycle
        distance_df.at[row_i, 'n_pulses'] = n_pulses

        row_i += 1

d_max = 500
d_width = 30

# Create bins
bins = np.arange(0, d_max + d_width, d_width)

# Assign each row in 'd' to a bin
distance_df['bin'] = pd.cut(distance_df['d'], bins=bins, right=False, include_lowest=True)



# Compute bin centers
bin_centers = bins[:-1] + d_width / 2  # Exclude the last bin edge and calculate centers

fig = utils.simple_fig(
    width=1, height=1
)

for dc, dcdf in distance_df.groupby('dc'):

    n_pulses = dcdf.iloc[0].n_pulses

    # Compute mean values of 'fr' for each bin
    mean_values = dcdf.groupby('bin')['fr'].mean().values

    fig.add_scatter(
        x=bin_centers, y=mean_values,
        mode='lines+markers', line=dict(color=clrs.duty_cycle(dc)),
        showlegend=True,
        name=f'{n_pulses:.0f}',
    )

fig.update_xaxes(
    tickvals=np.arange(0, 500, 50),
    title_text='distance from laser [um]'
)

fig.update_yaxes(
    tickvals=np.arange(0, 200, 50),
    title_text='fr [hz]'
)


sname = figure_dir / session_id / 'pa_dc_min_max' / f'fr_vs_distance_laser'
utils.save_fig(fig, sname, display=True)



#%% 3 MEA SCAN

trials = data_io.burst_df.query('protocol == "pa_mea_scan" and duty_cycle == 29')
burst_duration = trials.iloc[0].burst_duration
n_pulses = trials.iloc[0].burst_duration * trials.iloc[0].repetition_frequency / 1000
duty_cycle = trials.iloc[0].duty_cycle
print('MEA scan')
print(f'\nburst duration: {burst_duration:.0f} ms, n pulses: {n_pulses:.0f}, duty cycle: {duty_cycle:.0f}')


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
x_spacing = 0.001
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
    width=1,
    height=1.5,
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
        mode='markers', marker=dict(color='orange', size=4),
        showlegend=False,
        **pos,
    )

    fig.add_scatter(
        y=[yi], x=[xi],
        mode='markers', marker=dict(color='red', size=3),
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


#%% 3.1 Shift in centre of response

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

#%% 3.2 Distance between centre of response and laser

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


#%% 3.3A generate heatmaps

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
        fr_base = cells_df.loc[i, (train_id, 'baseline_firing_rate')]
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


# heatmap_df = heatmap_df.query('fr > 100')

x = heatmap_df.x_rel.values
y = heatmap_df.y_rel.values
z = heatmap_df.fr.values


grid_binwidth = 30
axlen = 300
xmin = -axlen
xmax = axlen
ymin = -axlen
ymax = axlen


n = int((xmax - xmin) / grid_binwidth)
xi = np.linspace(xmin, xmax, n+1)
yi = np.linspace(ymin, ymax, n+1)

# Create a grid for the heatmap
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
# zi = gaussian_filter(zi, sigma=3)

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
axtickstep = 100

for i in np.arange(-axlen, axlen, axtickstep):
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
    tickvals=np.arange(-axlen, axlen+axtickstep, axtickstep),
    range=[-axlen, axlen]
)


fig.update_yaxes(
    tickvals=np.arange(-axlen, axlen+axtickstep, axtickstep),
    range = [-axlen, axlen],
)


sname = figure_dir / session_id / 'mea_scan' / f'heatmap'
utils.save_fig(fig, sname, display=True)


#%% 3.3B heatmap firing rate relative to laser

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



#%% 4 TEMPORAL SERIES
# Gather clusters per electrode

trials = data_io.burst_df.query(f'protocol == "pa_temporal_series"')

clusters_per_electrode = {}
for electrode, df in trials.groupby('electrode'):

    cells = []

    for tid in df.train_id.unique():
        for cid in data_io.cluster_df.index.tolist():
            if cells_df.loc[cid, (tid, 'is_significant')] and cid not in cells:
                cells.append(cid)

    clusters_per_electrode[electrode] = cells


#%%

burst_duration = trials.iloc[0].burst_duration
n_pulses = trials.iloc[0].burst_duration * trials.iloc[0].repetition_frequency / 1000
duty_cycle = trials.iloc[0].duty_cycle
print('Temporal series')
print(f'\nburst duration: {burst_duration:.0f} ms, n pulses: {n_pulses:.0f}, duty cycle: {duty_cycle:.0f}')

print(f'series starts at {trials.burst_onset.min() / 1000:.0f} s')


#%% Define significance test to test single trial for significance
from scipy.stats import poisson

def poisson_test(n_baseline, t_baseline, n_stim, t_stim):
    # Baseline rate
    lambda_baseline = n_baseline / t_baseline

    # Compute p-value (one-tailed test)
    p_value = 1 - poisson.cdf(n_stim - 1, lambda_baseline * t_stim)
    return p_value


def detect_burst_significance(spiketrain, t_1):
    b_0 = -50
    b_1 = 0

    t_0 = 0
    # t_1 = 150
    bin_size = b_1 - b_0
    bin_hw = bin_size / 2
    t_step = 10

    bns = np.arange(t_0+bin_hw, t_1 - bin_hw, t_step)

    n_base = np.where((spiketrain >= b_0) & (spiketrain <= b_1))[0].size

    HAS_SIG_BIN = False
    for bin_i, bctr in enumerate(bns):
        t0 = bctr - bin_hw
        t1 = bctr + bin_hw
        n_stim = np.where((spiketrain >= t0) & (spiketrain < t1))[0].size

        p = poisson_test(n_base, bin_size / 1000, n_stim, bin_size / 1000)

        if p < 0.05:
            HAS_SIG_BIN = True
            break

    return HAS_SIG_BIN

#%% Plot spike train per cell and inter burst interval

n_cols = 3
n_rows = 2
x_domains = {}
y_domains = {}
x_offset = 0.05
x_spacing = 0.01
x_width = (1 - ((n_cols-1)*x_spacing) - 2 * x_offset) / n_cols
y_offset = 0.1
y_spacing = 0.15
y_height = (1 - ((n_rows - 1) * y_spacing) - 2 * y_offset) / n_rows
clrs = ProjectColors()

xmax = 200
xmin = -25

for row_i in range(n_rows):
    y1 = 1 - y_offset - row_i * (y_spacing + y_height)
    y_domains[row_i+1] = [[y1-y_height, y1] for _ in range(n_cols)]
    x_domains[row_i+1] = []
    for col_i in range(n_cols):
        x0 = x_offset + col_i * (x_spacing + x_width)
        x_domains[row_i+1].append([x0, x0+x_width])


for electrode in clusters_per_electrode.keys():
    trials_to_plot = trials.query(f'electrode == {electrode}')

    for cluster in clusters_per_electrode[electrode]:

        print(f'loading cluster: {cluster}')
        cluster_data = utils.load_obj(data_dir / 'bootstrapped' / f'bootstrap_{cluster}.pkl')

        fig = utils.make_figure(
            width=1,
            height=1.2,
            x_domains=x_domains,
            y_domains=y_domains,
            # equal_width_height='x',
            subplot_titles={
                1: [1000, 500, 250],
                2: [100, 50, 25],
            }
        )

        pos = {
            1000: dict(row=1, col=1),
            500: dict(row=1, col=2),
            250: dict(row=1, col=3),
            100: dict(row=2, col=1),
            50: dict(row=2, col=2),
            25: dict(row=2, col=3),
        }

        for tid, tdf in trials_to_plot.groupby('train_id'):
            train_period = tdf.iloc[0].train_period
            clr = clrs.train_period(train_period)
            p = pos[train_period]

            stimes = cluster_data[tid]['spike_times']

            x_plot, y_plot = [], []
            xp_sig, yp_sig = [], []

            for burst_i, sp in enumerate(stimes):
                sp_to_plot = sp[(sp > 0) & (sp < train_period)]
                x_plot.append(np.vstack([sp_to_plot, sp_to_plot, np.full(sp_to_plot.size, np.nan)]).T.flatten())
                y_plot.append(np.vstack([np.ones(sp_to_plot.size) * burst_i,
                                         np.ones(sp_to_plot.size) * burst_i + 1, np.full(
                        sp_to_plot.size, np.nan)]).T.flatten())
                is_sig = detect_burst_significance(sp, train_period)

                if is_sig:
                    xp_sig.append([xmin, xmin, xmax, xmax, None])
                    yp_sig.append([burst_i, burst_i + 1, burst_i + 1, burst_i, None])
                    n_sig += 1

            x_plot = np.hstack(x_plot)
            y_plot = np.hstack(y_plot)

            n_bursts = len(stimes)

            if train_period == 1000:
                perc_success = n_sig / n_bursts

            if len(xp_sig) > 0:
                xp_sig = np.hstack(xp_sig)
                yp_sig = np.hstack(yp_sig)

                fig.add_scatter(
                    x=xp_sig,
                    y=yp_sig,
                    mode='lines', line=dict(color='green', width=0),
                    fill='toself',
                    showlegend=False,
                    **p,
                )

            fig.add_scatter(
                x=x_plot, y=y_plot,
                mode='lines', line=dict(color=clr, width=0.5),
                showlegend=False,
                **p,
            )
            fig.update_xaxes(
                title_text='time [ms]' if p['row'] == 2 else '',
                tickvals=np.arange(-50, xmax + 50, 50),
                range=[xmin, xmax],
                **p
            )

            fig.update_yaxes(
                title_text='burst nr' if p['col'] == 1 else '',
                range=[0, len(stimes)],
                **p,
            )

        if perc_success > 0.7:
            subdir = 'sig'
        else:
            subdir = 'not_sig'
        sname = figure_dir / session_id / 'temporal_series' / subdir / f'{electrode:.0f}_{cluster}'
        utils.save_fig(fig, sname, display=False)
        print(f'saved')


#%% Plot raw trace following stimulus
from axorus.preprocessing.lib.filepaths import FilePaths
from axorus.preprocessing.params import nb_bytes_by_datapoint, data_nb_channels, data_sample_rate, dataset_dir, data_voltage_resolution, data_type
from scipy.signal import butter, filtfilt

# Define filter parameters
fs = 1000  # Sampling frequency in Hz (adjust to your actual sampling frequency)
cutoff = 50  # Cutoff frequency in Hz
order = 4  # Filter order

filepaths = FilePaths('241108_A', local_raw_dir=r'C:\Axorus\tmp2')

uid = 'uid_081124_001'
cluster_info = data_io.cluster_df.loc[uid]
cluster_channel = cluster_info.ch
bursts = data_io.burst_df.query('protocol == "pa_dc_min_max_series"')

to_plot = data_io.burst_df.query('electrode == 47')

for tid, df in to_plot.groupby('train_id'):
    # bursts = data_io.burst_df.query('protocol == "pa_temporal_series"')
    # rec_id = bursts.iloc[0].rec_id
    rec_id = df.iloc[0].rec_id

    # Retreive path to rawfile
    if filepaths.local_raw_dir is not None:
        input_file = filepaths.local_raw_dir / f'{rec_id}.raw'
    else:
        input_file = filepaths.raw_dir / f'{rec_id}.raw'

    # Open datafile
    m = np.memmap(input_file.as_posix(), dtype=data_type)

    cluster_channel = 47 - 1
    channel_index = np.arange(cluster_channel, m.size, data_nb_channels,
                              dtype=int)

    # Extract a burst
    burst_onset = df.iloc[2].burst_onset / 1000  # in [s]
    t_pre = 1  # [s]
    t_after = 1  # [s]
    n_pre = t_pre * data_sample_rate  # [samples]
    n_after = t_after * data_sample_rate  # [samples]
    n_samples = n_pre + n_after

    i_ref = burst_onset * data_sample_rate  # [samples]
    i0 = int(i_ref - n_pre)
    i1 = int(i_ref + n_after)

    burst_data = m[channel_index[i0:i1]]

    # Design the Butterworth highpass filter
    b, a = butter(order, cutoff, btype='high', fs=data_sample_rate)

    # Apply the filter to the data
    filtered_burst_data = filtfilt(b, a, burst_data)

    # General setup of figure
    n_rows = 1
    n_cols = 1

    x_domains = {}
    y_domains = {}
    subplot_titles = {}
    x_offset = 0.05
    y_offset = 0.1
    x_spacing = 0.1
    y_spacing = 0.1

    x_width = (1 - (n_cols-1) * x_spacing - 2 * x_offset) / n_cols
    y_height = (1 - (n_rows - 1) * y_spacing - 2 * y_offset) / n_rows

    for row_i in range(n_rows):
        x_domains[row_i+1] = []
        y_domains[row_i+1] = []
        subplot_titles[row_i+1] = []

        for col_j in range(n_cols):
            x0 = x_offset + col_j * (x_spacing + x_width)
            x_domains[row_i+1].append([x0, x0+x_width])

            y1 = 1 - y_offset - row_i * (y_spacing + y_height)
            y_domains[row_i+1].append([y1 - y_height, y1])


    y_min = np.min(burst_data)
    y_max = np.max(burst_data)

    fig = utils.make_figure(
        width=1, height=1.2,
        x_domains=x_domains,
        y_domains=y_domains,
        subplot_titles={1: [f'duty cycle:{df.iloc[0].duty_cycle:.0f}, '
                            f'train period: {df.iloc[0].train_period:.0f}']}
    )

    time = ((np.arange(i0, i1, 1) / data_sample_rate) - burst_onset ) * 1000

    fig.add_scatter(
        x=time, y=burst_data,
        mode='lines', line=dict(color='black', width=0.2),
        showlegend=False,
    )

    fig.add_scatter(
        x=[0, 0, 10, 10], y=[y_min, y_max, y_max, y_min],
        fillcolor='rgba(255, 0, 0, 0.3)', fill='toself',
        mode='lines', line=dict(width=0),
        showlegend=False,
    )

    fig.update_xaxes(
        range=[-100, 200],
        tickvals=np.arange(-1000, 1000, 100),
    )

    fig.update_yaxes(
        range=[y_min, y_max],
        tickvals=np.arange(y_min, y_max, 1000),
        ticktext=[f'{t:.0f}' for t in np.arange(y_min, y_max, 1000) / 1000],
    )

    sname = figure_dir / session_id / 'temporal_series' / 'raw_data' / (f'{uid}_duty_cycle_{df.iloc[0].duty_cycle:.0f}_'
                                                                        f'train_period_{df.iloc[0].train_period:.0f}_{tid}')
    utils.save_fig(fig, sname, display=False)


#%% 4 Epulse series

burst_df = data_io.burst_df.query('protocol == "pa_epulse"')

print('Np\t| Prf\t\t| Pe\t\t| Irradiance\t\t| burst duration')
for dc, dc_df in burst_df.groupby('n_turns'):
    e_pulse = dc_df.iloc[0].e_pulse
    frep = dc_df.iloc[0].repetition_frequency
    irr_exact = dc_df.iloc[0].irradiance_exact_fiber_diameter
    irr_min_3x = dc_df.iloc[0].irradiance_3x_fiber_diameter
    bd = dc_df.iloc[0].burst_duration
    npulses = (frep / 1000) * bd

    print(f'{npulses:.0f}\t| {frep:4.0f} Hz \t| {e_pulse:.2f} uJ\t| {irr_exact:.2f} - {irr_min_3x:.2f} W/mm2\t| {bd:.0f} ms')



#%% 4.1 single cell fr vs pulse energy

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

        e_pulse = []
        fr_baseline = []
        fr_response = []
        xticks = []

        for train_id, tdf in df.sort_values('e_pulse').groupby('train_id'):

            ep = tdf.iloc[0].e_pulse
            bd = tdf.iloc[0].burst_duration

            e_pulse.append(ep)
            xticks.append(f'{ep:.2f}')

            # rf.append(f'{tdf.iloc[0].repetition_frequency/1000:.1f}')
            # xticks.append(f'{tdf.iloc[0].repetition_frequency/1000:.1f}')

            fr_baseline.append(cells_df.loc[cluster_id, (train_id, 'baseline_firing_rate')])
            fr_response.append(cells_df.loc[cluster_id, (train_id, 'response_firing_rate')])

        # convert lists to numpy arrays
        e_pulse = np.array(e_pulse)
        fr_baseline = np.array(fr_baseline)
        fr_response = np.array(fr_response)
        xticks = np.array(xticks)

        # sort by dc values
        sort_idx = np.argsort(e_pulse)
        fr_baseline = fr_baseline[sort_idx]
        fr_response = fr_response[sort_idx]
        e_pulse = e_pulse[sort_idx]
        xticks = xticks[sort_idx]

        # Plot baseline data
        fig.add_scatter(
            x=e_pulse, y=fr_baseline,
            mode='lines+markers',
            line=dict(color='grey', width=1),
            showlegend=True if pos['row'] == 1 and pos['col'] == 1 else False,
            name='baseline',
            **pos,
        )

        # Plot response data
        fig.add_scatter(
            x=e_pulse, y=fr_response,
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
            title_text='pulse energy [uJ]',
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

    sname = figure_dir / session_id / 'pulse_e' / 'curve_per_cell' / cluster_id
    utils.save_fig(fig, sname, display=False)
