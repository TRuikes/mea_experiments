# %% General setup
import pandas as pd
from pathlib import Path
from axorus.preprocessing.project_colors import ProjectColors
from axorus.data_io import DataIO
import utils
import numpy as np
import axorus.analysis.figure_library as fl

# Load data
session_id = '250403_A'

# data_dir = Path(r'E:\Axorus\ex_vivo_series_3\dataset')
data_dir = Path(r'E:\Axorus\tmp')
figure_dir = Path(r'C:\Axorus\figures')
data_io = DataIO(data_dir)
loadname = data_dir / f'{session_id}_cells.csv'
data_io.load_session(session_id, load_pickle=True)
cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
clrs = ProjectColors()

INCLUDE_RANGE = 50  # include cells at max distance = 50 um

# blockers in this session
applied_blockers = ['noblocker', 'lap4', 'lap4acet', 'washout']


#%%


duty_cycles = [10, 20, 30]
stim_sites = [116, 132, 180]


burst_df = data_io.burst_df.query('blockers == "noblocker"')

for dc in duty_cycles:
    df = burst_df.query(f'duty_cycle == {dc}')

    train_ids = df.train_id.unique()

    n_sig = 0
    for tid in train_ids:
        n_sig += cells_df[tid, 'is_significant'].sum()

    print(dc, n_sig)


#%%


for ec in stim_sites:
    # ec = stim_sites[0]
    # dc = 30
    # tid = data_io.burst_df.query(f'blockers == "noblocker" and electrode == {ec}'
    #                              f'and duty_cycle == {dc}').train_id.unique()[0]


    # sig_cells = cells_df.loc[cells_df[tid, 'is_significant'] == True]

    for cluster_id in cells_df.index.values:
    # cluster_id = sig_cells.index.values[5]

        fig = utils.make_figure(
            width=1,
            height=1.5,
            x_domains={
                1: [[0.1, 0.9],],
            },
            y_domains={
                1: [[0.1, 0.9],]
            },
            subplot_titles={
                1: ['']
            }
        )

        x_plot, y_plot, clr = [], [], []
        burst_offset = 0

        cluster_data = utils.load_obj(data_dir / 'bootstrapped' / f'bootstrap_{cluster_id}.pkl')

        yticks = []
        ytext = []

        for blocker in applied_blockers:
            for dc in duty_cycles:
                tid = data_io.burst_df.query(f'blockers == "{blocker}" and electrode == {ec}'
                                             f'and duty_cycle == {dc}').train_id.unique()[0]
                spike_times = cluster_data[tid]['spike_times']
                bins = cluster_data[tid]['bins']

                # ytext.append(f'Pr: {frep/1000:.1f} [kHz]')
                # yticks.append(burst_offset + len(spike_times) / 2)

                for burst_i, sp in enumerate(spike_times):
                    # ytext.append(data_io.burst_df.query('train_id == @tid').iloc[burst_i].name.split('_')[-1])
                    # yticks.append(burst_offset + 1)
                    x_plot.append(np.vstack([sp, sp, np.full(sp.size, np.nan)]).T.flatten())
                    y_plot.append(np.vstack([np.ones(sp.size) * burst_i + burst_offset,
                                             np.ones(sp.size) * burst_i + 1 + burst_offset, np.full(sp.size, np.nan)]).T.flatten())

                    ytext.append(f'{blocker} {dc:.0f}')
                    yticks.append(burst_offset + len(spike_times) / 2)

                        # if blocker == 'noblocker':
                    #     clr.extend(['black' for _ in range(sp.size * 3)])
                    # elif blocker == 'lap4':
                    #     clr.append(['orange'  for _ in range(sp.size * 3)])
                    # elif blocker == 'lap4acet':
                    #     clr.append(['red' for _ in range(sp.size * 3)])
                    # elif blocker == 'washout':
                    #     clr.append(['blue'  for _ in range(sp.size * 3)])

                burst_offset += len(spike_times)

        x_plot = np.hstack(x_plot)
        y_plot = np.hstack(y_plot)

        fig.add_scatter(
            x=x_plot, y=y_plot,
            mode='lines', line=dict(color='black', width=0.5),
            showlegend=False,
            # **pos,
        )

        fig.update_xaxes(
            tickvals=np.arange(-500, 500, 100),
            title_text=f'time [ms]',
            range=[bins[0] - 1, bins[-1] + 1],
            # **pos,
        )

        fig.update_yaxes(
            # range=[0, n_bursts],
            tickvals=yticks,
            ticktext=ytext,
            # **pos,
        )

        sname = figure_dir / session_id / 'rasterplot' / f'{ec:.0f}' / f'{cluster_id}'
        utils.save_fig(fig, sname, display=False)



