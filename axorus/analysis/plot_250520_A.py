# %% General setup
import pandas as pd
from pathlib import Path
from axorus.data_io import DataIO
import utils
import numpy as np
from utils import make_figure, save_fig
from scipy.stats import wilcoxon
from axorus.preprocessing.project_colors import ProjectColors

# Load data
session_id = '250520_A'
data_dir = Path(r'E:\Axorus\dataset_series_3')
figure_dir = Path(r'E:\Axorus\Figures') / 'lap4analysis'
data_io = DataIO(data_dir)
loadname = data_dir / f'{session_id}_cells.csv'
data_io.load_session(session_id, load_pickle=True)
cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
clrs = ProjectColors()

INCLUDE_RANGE = 50  # include cells at max distance = 50 um

clrs = ProjectColors()


#%% Detect electrode stim site with most significant responses, per cell

electrodes = [[164], [155], [63], [217, 100]]

pref_ec_dict = {}

for cluster_id in data_io.cluster_df.index.values:

    pref_ec = None
    n_sig_pref_ec = None

    max_fr = None
    for ec in electrodes:
        df = data_io.burst_df.query(f'electrode in {ec} and blockers == "noblocker"')
        tids = df.train_id.unique()
        n_sig = 0
        for tid in tids:
            if cells_df.loc[cluster_id, (tid, 'is_significant')] is True:
                n_sig += 1

        if n_sig > 1:
            if pref_ec is None or n_sig > n_sig_pref_ec:
                pref_ec = ec
                n_sig_pref_ec = n_sig

    pref_ec_dict[cluster_id] = pref_ec


#%% Plot raster plots for each individual cell

cluster_ids = data_io.cluster_df.index.values
electrodes = data_io.burst_df.electrode.unique()

blockers = ['noblocker', 'lap4', 'lap4acet', 'washout']

for cluster_id in cluster_ids:

    cluster_data = utils.load_obj(data_dir / 'bootstrapped' / f'bootstrap_{cluster_id}.pkl')

    n_electrodes = electrodes.size

    electrode = pref_ec_dict[cluster_id]
    if electrode is None:
        continue

    fig = utils.make_figure(
        width=1,
        height=1.5,
        x_domains={
            1: [[0.1, 0.9]],
        },
        y_domains={
            1: [[0.1, 0.9]]
        },
    )

    burst_offset = 0
    x_plot, y_plot = [], []
    yticks = []
    ytext = []
    pos = dict(row=1, col=1)

    has_sig = False

    for blocker in blockers:
        d_select = data_io.burst_df.query('electrode in @electrode and '
                                              'blockers == @blocker').copy()
        d_select.sort_values('repetition_frequency', inplace=True)
        repetition_frequencies = d_select.repetition_frequency.unique()

        for rf in repetition_frequencies:
            tid = d_select.query('repetition_frequency == @rf').iloc[0].train_id

            frep = data_io.burst_df.query('train_id == @tid').iloc[0].repetition_frequency
            spike_times = cluster_data[tid]['spike_times']
            bins = cluster_data[tid]['bins']

            ytext.append(f'Pr: {frep/1000:.1f} [kHz], {blocker}')
            yticks.append(burst_offset + len(spike_times) / 2)

            for burst_i, sp in enumerate(spike_times):
                x_plot.append(np.vstack([sp, sp, np.full(sp.size, np.nan)]).T.flatten())
                y_plot.append(np.vstack([np.ones(sp.size) * burst_offset,
                                         np.ones(sp.size)* burst_offset +1, np.full(sp.size, np.nan)]).T.flatten())
                burst_offset += 1

    x_plot = np.hstack(x_plot)
    y_plot = np.hstack(y_plot)

    fig.add_scatter(
        x=x_plot, y=y_plot,
        mode='lines', line=dict(color='black', width=0.5),
        showlegend=False,
        **pos,
    )

    fig.update_xaxes(
        tickvals=np.arange(-500, 500, 100),
        title_text=f'time [ms]',
        range=[bins[0]-1, bins[-1]+1],
        **pos,
    )

    fig.update_yaxes(
        # range=[0, n_bursts],
        tickvals=yticks,
        ticktext=ytext,
        **pos,
    )

    sname = figure_dir /  session_id / 'raster plots' / f'{cluster_id}'

    utils.save_fig(fig, sname, display=False)


#%% plot individual firing rates
cluster_ids = data_io.cluster_df.index.values
electrodes = data_io.burst_df.electrode.unique()

blockers = ['noblocker', 'lap4', 'lap4acet', 'washout']

for cluster_id in cluster_ids:

    cluster_data = utils.load_obj(data_dir / 'bootstrapped' / f'bootstrap_{cluster_id}.pkl')

    n_electrodes = electrodes.size

    electrode = pref_ec_dict[cluster_id]
    if electrode is None:
        continue

    fig = utils.make_figure(
        width=1,
        height=1.5,
        x_domains={
            1: [[0.1, 0.9]],
        },
        y_domains={
            1: [[0.1, 0.9]]
        },
    )

    burst_offset = 0
    x_plot, y_plot = [], []
    yticks = []
    ytext = []
    pos = dict(row=1, col=1)

    has_sig = False

    for blocker in blockers:
        d_select = data_io.burst_df.query('electrode in @electrode and '
                                              'blockers == @blocker').copy()
        d_select.sort_values('repetition_frequency', inplace=True)
        repetition_frequencies = d_select.repetition_frequency.unique()

        fmax = np.max(repetition_frequencies)

        tid = d_select.query('repetition_frequency == @fmax').iloc[0].train_id

        frep = data_io.burst_df.query('train_id == @tid').iloc[0].repetition_frequency
        bins = cluster_data[tid]['bins']
        fr = cluster_data[tid]['firing_rate']
        fr_ci_low = cluster_data[tid]['firing_rate_ci_low']
        fr_ci_high = cluster_data[tid]['firing_rate_ci_high']

        if 'noblocker' in blocker:
            clr = clrs.blocker_color('none', 1)
            clr_a = clrs.blocker_color('none', 0.1)

        elif 'lap4' in blocker and 'acet' not in blocker:
            clr = clrs.blocker_color('lap4', 1)
            clr_a = clrs.blocker_color('lap4', 0.1)

        elif 'lap4' in blocker and 'acet' in blocker:
            clr = clrs.blocker_color('lap4acet', 1)
            clr_a = clrs.blocker_color('lap4acet', 0.1)

        elif 'washout' in blocker:
            clr = clrs.blocker_color('washout', 1)
            clr_a = clrs.blocker_color('washout', 0.1)

        else:
            clr = None
            clr_a = None

        fig.add_scatter(
            x=bins, y=fr_ci_low,
            mode='lines', line=dict(color=clr_a, width=0),
            showlegend=False,
            name=blocker,
            **pos,
        )
        fig.add_scatter(
            x=bins, y=fr_ci_high,
            mode='lines', line=dict(color=clr_a, width=0),
            showlegend=False,
            name=blocker,
            fill='tonexty',
            **pos,
        )

        fig.add_scatter(
            x=bins, y=fr,
            mode='lines', line=dict(color=clr, width=0.5),
            showlegend=True,
            name=blocker,
            **pos,
        )

    fig.update_xaxes(
        tickvals=np.arange(-500, 500, 100),
        title_text=f'time [ms]',
        range=[bins[0]-1, bins[-1]+1],
        **pos,
    )

    fig.update_yaxes(
        # range=[0, n_bursts],
        tickvals=np.arange(0, 300, 30),
        title_text=f'firing rate [Hz]',
        **pos,
    )

    sname = figure_dir / session_id / 'firing rate plots' / f'{cluster_id}'

    utils.save_fig(fig, sname, display=False)





#%% Gather data for final plot

blockers = ['noblocker', 'lap4', 'lap4acet', 'washout']
df_plot = pd.DataFrame()

for cluster_id, electrode in pref_ec_dict.items():
    if electrode is None:
        continue

    for blocker in blockers:
        d_select = data_io.burst_df.query('electrode in @electrode and '
                                              'blockers == @blocker').copy()
        tid = d_select.loc[d_select['repetition_frequency'].idxmax()].train_id

        df_plot.at[cluster_id, f'{blocker} baseline'] = cells_df.loc[cluster_id, tid].baseline_firing_rate
        df_plot.at[cluster_id, f'{blocker} response'] = cells_df.loc[cluster_id, tid].response_firing_rate


def print_wilcoxon(d0, d1, tag1, tag2):

    idx = np.where(pd.notna(d0) & pd.notna(d1))[0]

    r, p = wilcoxon(d0[idx], d1[idx])
    d0_m = np.mean(d0[idx])
    d0_s = np.std(d0[idx])
    d1_m = np.mean(d1[idx])
    d1_s = np.std(d1[idx])
    print(f'{tag1} vs {tag2}:({d0_m:.0f} ({d0_s:.0f}), {d1_m:.0f} ({d1_s:.0f})) T = {r:.0f} (p={p:.3f})')

# ## FIGURE SETUP
n_rows = 1
n_cols = 1

y_top = 0.1
y_bottom = 0.1
yspacing = 0.1  # y space between rows

xoffset = 0.1  # x space left and right of plots
xspacing = 0.1  # x space between columns

yheight = (1 - y_bottom - y_top - yspacing * (n_rows - 1))  # height of each plot
rel_heights = [1, 1, 1]

# Generate x and y spacing for all the subplots
y_domains = dict()
y1 = 1 - y_top
for i in range(n_rows):
    row_h = yheight * rel_heights[i]
    y0 = y1 - row_h
    y_domains[i + 1] = [[y0, y1] for j in range(n_cols)]

    y1 -= (row_h + yspacing)

xwidth = (1 - (n_cols - 1) * xoffset - xspacing - 0.05) / n_cols
sx = [[xoffset + (xspacing + xwidth) * i, xoffset + (xspacing + xwidth) * i + xwidth] for i in range(n_cols)]
clrs = ProjectColors()

# Generate the figure
fig = make_figure(
    width=0.3, height=0.6,
    x_domains={
        1: sx,
    },
    y_domains=y_domains,
    subplot_titles={
        1: ['', '', ],
    },
)

xpos = [0, 1, 3, 4, 6, 7, 9, 10]
xdata = ['noblocker baseline', 'noblocker response',
         'lap4 baseline', 'lap4 response',
         'lap4acet baseline', 'lap4acet response',
         'washout baseline', 'washout response']

xlbl = ['no blocker', 'lap4', 'lap4+acet', 'washout']

n_pts = df_plot.shape[0]

# n_retinas = len(list(P23H_data.keys()))f
print(f'CPP experiments')
print(f'{n_pts} cells, {1} retinas')

# print_wilcoxon(responses['baseline_none'], responses['baseline_cpp'], 'baseline none', 'baseline cpp')
# print_wilcoxon(responses['baseline_cpp'], responses['stim_cpp'], 'baseline cpp', 'stim cpp')
# print_wilcoxon(responses['stim_none'], responses['stim_cpp'], 'stim none', 'stim cpp')
# print_wilcoxon(responses['stim_none'], responses['stim_wash'], 'stim none', 'stim wash')
print_wilcoxon(df_plot['noblocker baseline'], df_plot['noblocker response'], 'none baseline', 'none stim')
print_wilcoxon(df_plot['lap4 baseline'], df_plot['lap4 response'], 'lap4 baseline', 'stim lap4')
print_wilcoxon(df_plot['lap4acet baseline'], df_plot['lap4acet response'], 'lap4acet baseline', 'lap4acet stim')
print_wilcoxon(df_plot['washout baseline'], df_plot['washout response'], 'washout baseline', 'washout stim')


box_specs = dict(
    name='P23H',
    boxpoints='all',
    marker=dict(color=clrs.animal_color('LE', 1, 1), size=2),
    line=dict(color=clrs.animal_color('LE', 1, 1), width=1.5),
    showlegend=False,
)

for xp, xd in zip(xpos, xdata):
    fig.add_box(
        x=np.ones(n_pts) * xp,
        y=df_plot[xd].values,
        **box_specs,
    )

fig.update_yaxes(
    range=[0, 250],
    title_text=f'fr. [Hz]',
    tickvals=np.arange(0, 300, 50),
)
fig.update_xaxes(
    tickvals=[0.5, 3.5, 6.5, 9.5],
    ticktext=['no blocker', 'lap4', 'lap4acet', 'washout'],
)

sname = figure_dir / f'{session_id}_boxplot'
save_fig(fig, sname, formats=['png', 'svg'], scale=3)
