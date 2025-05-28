# %% General setup
import pandas as pd
from pathlib import Path
from axorus.preprocessing.project_colors import ProjectColors
from axorus.data_io import DataIO
import utils
import numpy as np
import axorus.analysis.figure_library as fl

# Load data
session_id = '250520_A'
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

cluster_ids = data_io.cluster_df.index.values
electrodes = data_io.burst_df.electrode.unique()

blockers = ['noblocker', 'lap4', 'lap4acet', 'washout']

for cluster_id in cluster_ids:

    cluster_data = utils.load_obj(data_dir / 'bootstrapped' / f'bootstrap_{cluster_id}.pkl')

    n_electrodes = electrodes.size

    for electrode in electrodes:

        if electrode == 217:
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

        bursts = data_io.burst_df.query('electrode == @electrode')
        trains = bursts.train_id.unique()
        duty_cycles = bursts.duty_cycle.unique()

        n_trains = trains.size

        burst_offset = 0
        x_plot, y_plot = [], []
        yticks = []
        ytext = []
        pos = dict(row=1, col=1)

        has_sig = False

        for blocker in blockers:
            if electrode == 100:
                ecs = [100, 217]
            else:
                ecs = [electrode]
            d_select = data_io.burst_df.query('electrode in @ecs and '
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

                if cluster_data[tid]['is_sig']:
                    has_sig = True

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

                # # Plot shaded areas for significance
                # burst_offset = 0
                # for tid in train_ids:
                #
                #     spike_times = cluster_data[tid]['spike_times']
                #     is_sig = cluster_data[tid]['is_sig']
                #     nb = len(spike_times)
                #
                #     clr = 'rgba(0, 255, 0, 0.3)' if is_sig else 'rgba(255, 0, 0, 0.3)'
                #     fig.add_scatter(
                #         x=[0, 0, 20, 20],
                #         y=[burst_offset, burst_offset + nb, burst_offset + nb, burst_offset],
                #         mode='lines', line=dict(color=clr, width=0.1),
                #         fill='toself', fillcolor=clr,
                #         showlegend=False,
                #         **pos,
                #     )
                #     burst_offset += nb
                #
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

        if not has_sig:
            sname = figure_dir / 'rasterplot_per_laser_level' / session_id / 'not_sig' / f'{electrode:.0f}' / f'{cluster_id}'
        else:
            sname = figure_dir / 'rasterplot_per_laser_level' / session_id / f'{electrode:.0f}' / f'{cluster_id}'

        utils.save_fig(fig, sname, display=False)

#%%

# for cluster_id in cluster_ids:
#     for electrode in electrodes:
#
#         train_ids = data_io.burst_df.query('electrode == @electrode').train_id.unique()
#
#         data_to_plot = pd.DataFrame()
#         for tid in train_ids:
#             data_to_plot.at[tid, 'frep'] = data_io.burst_df.query('train_id == @tid').iloc[0].repetition_frequency
#             data_to_plot.at[tid, 'e_pulse'] = data_io.burst_df.query('train_id == @tid').iloc[0].e_pulse
#             data_to_plot.at[tid, 'irr'] = data_io.burst_df.query('train_id == @tid').iloc[0].irradiance
#             data_to_plot.at[tid, 'frate'] = df.loc[cluster_id, (tid, 'response_firing_rate')]
#             data_to_plot.at[tid, 'clr'] = 'green' if df.loc[cluster_id, (tid, 'is_significant')] else 'red'
#
#         y_offset = 0.1
#         n_y = 3
#         y_spacing = 0.15
#         y_height = (1 - (2*y_offset) - (n_y-1) * y_spacing) / n_y
#         y_domains = {}
#         for i in range(n_y):
#             y0 = 1 - y_offset - i * (y_height + y_spacing)
#             y1 = y0 - y_height
#             y_domains[i+1] = [[y1, y0]]
#
#         fig = utils.make_figure(
#             width=0.5,
#             height=1.2,
#             x_domains={
#                 1: [[0.1, 0.9]],
#                 2: [[0.1, 0.9]],
#                 3: [[0.1, 0.9]],
#
#             },
#             y_domains=y_domains,
#         )
#
#         # Plot firing rate as function of energy per pulse
#         for plot_row, data in zip([1, 2, 3], ['frep', 'e_pulse', 'irr']):
#             pos = dict(row=plot_row, col=1)
#
#             x = data_to_plot[data].values
#             y = data_to_plot['frate'].values
#             clr = data_to_plot['clr'].values
#
#             fig.add_scatter(
#                 x=x, y=y,
#                 mode='markers',
#                 line=dict(color='black', width=1),
#                 marker=dict(color=clr, size=4),
#                 showlegend=False,
#                 **pos,
#             )
#
#             if data == 'frep':
#                 xticks = np.arange(8000, 16000, 1000)
#                 xtext = np.arange(8, 16, 1)
#                 xmin = 8000
#                 xmax = 16000
#                 xtitle = 'rep. freq. [kHz]'
#             elif data == 'e_pulse':
#                 xticks = np.arange(1.6, 2, 0.1)
#                 xtext = [f'{x:.1f}' for x in xticks]
#                 xmin = 1.6
#                 xmax = 2
#                 xtitle = 'Epulse [uJ]'
#             elif data == 'irr':
#                 xticks = np.arange(8, 13.5, 0.5)
#                 xtext = xticks
#                 xmin = 8
#                 xmax = 13
#                 xtitle = 'irradiance [mW/mm2]'
#
#             fig.update_xaxes(
#                 range=[xmin, xmax],
#                 tickvals=xticks,
#                 ticktext=xtext,
#                 title_text=f'{xtitle}',
#                 **pos,
#             )
#
#             y0 = np.nanmin(y)
#             y1 = np.nanmax(y)
#             fig.update_yaxes(
#                 tickvals=np.arange(0, 100, 10),
#                 range=[y0-2, y1+2],
#                 title_text=f'Frate [Hz]',
#                 **pos,
#             )
#
#         sname = figure_dir / 'firing_rate_per_param' / f'{electrode}' / f'{cluster_id}'
#         utils.save_fig(fig, sname, display=False)
