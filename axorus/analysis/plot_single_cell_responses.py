import pandas as pd
from pathlib import Path
from axorus.data_io import DataIO
import utils
import numpy as np
sessions = ('161024_A',)

data_dir = Path(r'F:\thijs\series_3\dataset')
figure_dir = Path(r'E:\Axorus\figures')
data_io = DataIO(data_dir)
session_id = sessions[0]
loadname = data_dir / f'{session_id}_cells.csv'

data_io.load_session(session_id)

df = pd.read_csv(loadname, header=[0, 1], index_col=0)


#%%

cluster_ids = data_io.cluster_df.index.values
electrodes = data_io.train_df.electrode.unique()

for cluster_id in cluster_ids:

    cluster_data = utils.load_obj(data_dir / 'bootstrapped' / f'bootstrap_{cluster_id}.pkl')

    n_electrodes = electrodes.size

    for electrode in electrodes:

        bursts = data_io.train_df.query('electrode == @electrode')
        trains = bursts.train_id.unique()
        laser_levels = bursts.laser_level.unique()

        n_trains = trains.size

        print(f'electrode: {electrode:.0f} has {n_trains} trains, recorded in {laser_levels.size} laser levels')

        fig = utils.make_figure(
            width=1,
            height=1.5,
            x_domains={
                1: [[0.1, 0.4], [0.6, 0.9]],
            },
            y_domains={
                1: [[0.1, 0.9], [0.1, 0.9]]
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
                freps.append(data_io.train_df.query('train_id == @tid').iloc[0].repetition_frequency)
            train_ids = train_ids[np.argsort(freps)[::-1]]

            # Plot spikes
            for tid in train_ids:

                frep = data_io.train_df.query('train_id == @tid').iloc[0].repetition_frequency
                spike_times = cluster_data[tid]['spike_times']
                bins = cluster_data[tid]['bins']

                ytext.append(f'Fr: {frep/1000:.1f} [kHz]')
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
        utils.save_fig(fig, sname, display=True)

        break
    break

#%%

for cluster_id in cluster_ids:
    for electrode in electrodes:

        train_ids = data_io.train_df.query('electrode == @electrode').train_id.unique()

        data_to_plot = pd.DataFrame()
        for tid in train_ids:
            data_to_plot.at[tid, 'frep'] = data_io.train_df.query('train_id == @tid').iloc[0].repetition_frequency
            data_to_plot.at[tid, 'e_pulse'] = data_io.train_df.query('train_id == @tid').iloc[0].e_pulse
            data_to_plot.at[tid, 'irr'] = data_io.train_df.query('train_id == @tid').iloc[0].irradiance
            data_to_plot.at[tid, 'frate'] = df.loc[cluster_id, (tid, 'response_firing_rate')]
            data_to_plot.at[tid, 'clr'] = 'green' if df.loc[cluster_id, (tid, 'is_significant')] else 'red'

        y_offset = 0.1
        n_y = 3
        y_spacing = 0.15
        y_height = (1 - (2*y_offset) - (n_y-1) * y_spacing) / n_y
        y_domains = {}
        for i in range(n_y):
            y0 = 1 - y_offset - i * (y_height + y_spacing)
            y1 = y0 - y_height
            y_domains[i+1] = [[y1, y0]]

        fig = utils.make_figure(
            width=0.5,
            height=1.2,
            x_domains={
                1: [[0.1, 0.9]],
                2: [[0.1, 0.9]],
                3: [[0.1, 0.9]],

            },
            y_domains=y_domains,
        )

        # Plot firing rate as function of energy per pulse
        for plot_row, data in zip([1, 2, 3], ['frep', 'e_pulse', 'irr']):
            pos = dict(row=plot_row, col=1)

            x = data_to_plot[data].values
            y = data_to_plot['frate'].values
            clr = data_to_plot['clr'].values

            fig.add_scatter(
                x=x, y=y,
                mode='markers',
                line=dict(color='black', width=1),
                marker=dict(color=clr, size=4),
                showlegend=False,
                **pos,
            )

            if data == 'frep':
                xticks = np.arange(8000, 16000, 1000)
                xtext = np.arange(8, 16, 1)
                xmin = 8000
                xmax = 16000
                xtitle = 'rep. freq. [kHz]'
            elif data == 'e_pulse':
                xticks = np.arange(1.6, 2, 0.1)
                xtext = [f'{x:.1f}' for x in xticks]
                xmin = 1.6
                xmax = 2
                xtitle = 'Epulse [uJ]'
            elif data == 'irr':
                xticks = np.arange(8, 13.5, 0.5)
                xtext = xticks
                xmin = 8
                xmax = 13
                xtitle = 'irradiance [mW/mm2]'

            fig.update_xaxes(
                range=[xmin, xmax],
                tickvals=xticks,
                ticktext=xtext,
                title_text=f'{xtitle}',
                **pos,
            )

            y0 = np.nanmin(y)
            y1 = np.nanmax(y)
            fig.update_yaxes(
                tickvals=np.arange(0, 100, 10),
                range=[y0-2, y1+2],
                title_text=f'Frate [Hz]',
                **pos,
            )

        sname = figure_dir / 'firing_rate_per_param' / f'{electrode}' / f'{cluster_id}'
        utils.save_fig(fig, sname, display=False)
