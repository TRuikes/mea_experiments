import pandas as pd
from pathlib import Path
from axorus.data_io import DataIO
import utils
import numpy as np

session_id = '241024_A'
data_dir = Path(r'D:\Axorus\ex_vivo_series_3\dataset')
figure_dir = Path(r'C:\Axorus\figures')
data_io = DataIO(data_dir)
loadname = data_dir / f'{session_id}_cells.csv'

data_io.load_session(session_id)

df = pd.read_csv(loadname, header=[0, 1], index_col=0)

#%% Plot the DC series

rec_to_plot = data_io.burst_df.query('recording_name == "241024_A_1_noblocker"')
cluster_ids = data_io.cluster_df.index.values
electrodes = rec_to_plot.electrode.unique()

for cluster_id in cluster_ids:
    cluster_data = utils.load_obj(data_dir / 'bootstrapped' / f'bootstrap_{cluster_id}.pkl')

    n_electrodes = electrodes.size

    for electrode in electrodes:

        bursts = data_io.burst_df.query('electrode == @electrode')
        trains = bursts.train_id.unique()
        laser_levels = bursts.laser_level.unique()

        n_trains = trains.size
        print(f'electrode: {electrode:.0f} has {n_trains} trains, recorded in {laser_levels.size} laser levels')

        fig = utils.make_figure(
            width=1,
            height=1.5,
            x_domains={
                1: [[0.1, 0.3], [0.4, 0.6], [0.7, 0.9]],
            },
            y_domains={
                1: [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]
            },
            subplot_titles={
                1: [f'laser level {l:.0f} %' for l in laser_levels]
            }
        )

  # Count the nr of bursts per laser level
        n_bursts = 0
        for laser_level in laser_levels:
            d_select = bursts.query('laser_level == @laser_level')
            n = d_select.shape[0]
            if n > n_bursts:
                n_bursts = n

        print(f'max burst per laser level: {n_bursts}')

        for laser_i, laser_level in enumerate(laser_levels):
            pos = dict(row=1, col=laser_i+1)

            d_select = bursts.query('laser_level == @laser_level')
            train_ids = d_select.train_id.unique()

            x_plot, y_plot = [], []
            burst_offset = 0

            # Plot reference at 0
            fig.add_scatter(
                x=[0, 0],
                y=[0, n_bursts],
                mode='lines',
                line=dict(color='red', width=1),
                showlegend=False,
                **pos,
            )

            yticks = []
            ytext = []

            # Sort train ids by descending repetition frequency
            freps = []
            for tid in train_ids:
                freps.append(data_io.burst_df.query('train_id == @tid').iloc[0].repetition_frequency)
            train_ids = train_ids[np.argsort(freps)[::-1]]

            # Plot spikes
            for tid in train_ids:

                frep = data_io.burst_df.query('train_id == @tid').iloc[0].repetition_frequency
                spike_times = cluster_data[tid]['spike_times']
                bins = cluster_data[tid]['bins']

                ytext.append(f'Pr: {frep/1000:.1f} [kHz]')
                yticks.append(burst_offset + len(spike_times) / 2)

                for burst_i, sp in enumerate(spike_times):
                    x_plot.append(np.vstack([sp, sp, np.full(sp.size, np.nan)]).T.flatten())
                    y_plot.append(np.vstack([np.ones(sp.size)*burst_i + burst_offset,
                                             np.ones(sp.size)*burst_i+1 + burst_offset, np.full(sp.size, np.nan)]).T.flatten())

                burst_offset += len(spike_times)

            x_plot = np.hstack(x_plot)
            y_plot = np.hstack(y_plot)

            fig.add_scatter(
                x=x_plot, y=y_plot,
                mode='lines', line=dict(color='black', width=0.5),
                showlegend=False,
                **pos,
            )

            # Plot shaded areas for significance
            burst_offset = 0
            for tid in train_ids:

                spike_times = cluster_data[tid]['spike_times']
                is_sig = cluster_data[tid]['is_sig']
                nb = len(spike_times)

                clr = 'rgba(0, 255, 0, 0.3)' if is_sig else 'rgba(255, 0, 0, 0.3)'
                fig.add_scatter(
                    x=[0, 0, 20, 20],
                    y=[burst_offset, burst_offset + nb, burst_offset + nb, burst_offset],
                    mode='lines', line=dict(color=clr, width=0.1),
                    fill='toself', fillcolor=clr,
                    showlegend=False,
                    **pos,
                )
                burst_offset += nb

            fig.update_xaxes(
                tickvals=np.arange(-500, 500, 100),
                title_text=f'time [ms]',
                range=[bins[0]-1, bins[-1]+1],
                **pos,
            )

            fig.update_yaxes(
                range=[0, n_bursts],
                tickvals=yticks,
                ticktext=ytext,
                **pos,
            )

        sname = figure_dir / 'rasterplot_per_laser_level' / f'{electrode}' / f'{cluster_id}'
        utils.save_fig(fig, sname, display=False)
