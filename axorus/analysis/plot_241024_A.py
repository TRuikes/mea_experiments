import pandas as pd
from pathlib import Path
from axorus.data_io import DataIO
import utils
import numpy as np
from axorus.preprocessing.params import data_sample_rate, data_type, data_nb_channels

session_id = '241024_A'
data_dir = Path(r'F:\Axorus\ex_vivo_series_3\dataset')
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

                # ytext.append(f'Pr: {frep/1000:.1f} [kHz]')
                # yticks.append(burst_offset + len(spike_times) / 2)

                for burst_i, sp in enumerate(spike_times):
                    ytext.append(data_io.burst_df.query('train_id == @tid').iloc[burst_i].name.split('_')[-1])
                    yticks.append(burst_offset+1)
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

        sname = figure_dir / session_id / 'rasterplot_per_laser_level' / f'{electrode}' / f'{cluster_id}'
        utils.save_fig(fig, sname, display=False)


#%% Plot waveforms

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

#%% Plot raw traces from single trials
from axorus.preprocessing.lib.filepaths import FilePaths
from phylib.io.model import load_model
from axorus.preprocessing.project_colors import ProjectColors

clrs = ProjectColors()

filepaths = FilePaths(session_id)
model = load_model(filepaths.proc_sc_params)
cluster_overview = pd.read_csv(filepaths.proc_phy_cluster_info, sep='\t', header=0, index_col=0)

channels_per_clusters = {}
for i, r in data_io.cluster_df.iterrows():
    chs = model.get_cluster_channels(r.phy_cluster_id)
    channels_per_clusters[i] = chs

#%%

import numpy as np
from scipy.signal import medfilt
def highpass_filter_convolution(x, y, cutoff_freq):
    dt = x[1] - x[0]
    nyquist_freq = 0.5 / dt
    normal_cutoff = cutoff_freq / nyquist_freq

    n = int(1 / normal_cutoff)
    n = n if n % 2 else n + 1  # Ensure odd number for kernel symmetry

    # Create a highpass filter kernel
    kernel = -np.ones(n) / n
    kernel[n // 2] += 1

    # Apply convolution
    y_filtered = np.convolve(y, kernel, mode='same')
    return y_filtered



#%%
for channel_nr in data_io.cluster_df.ch.unique():
# for channel_nr in [2]:
    bursts = data_io.burst_df.query('laser_level == 65')
    n_to_print = 5
    onsets = bursts.burst_onset.values
    t0 = -200
    t1 = 300
    n = 0

    for bid, binfo in bursts.iterrows():
        recname = binfo.recording_name

        input_file = filepaths.raw_dir / f'{recname}.raw'
        m = np.memmap(input_file.as_posix(), dtype=data_type)

        channel_index = np.arange(channel_nr, m.size, data_nb_channels, dtype=int)
        channel_time = np.arange(0, channel_index.size, 1) / 20

        idx = np.argwhere((channel_time >= binfo.burst_onset + t0) & (channel_time <= binfo.burst_onset + t1))
        burst_data = m[channel_index[idx.flatten()]]
        burst_time = channel_time[idx.flatten()] - binfo.burst_onset

        fig = utils.make_figure(
            width=1,
            height=0.8,
            x_domains={1: [[0.1, 0.9]]},
            y_domains={1: [[0.1, 0.9]]},
        )
        fig.add_scatter(
            x=burst_time, y=burst_data,
            mode='lines', line=dict(color='black', width=0.4),
            showlegend=False,
        )

        # detect which clusters have spikes on this channel
        uids_to_plot = []
        for uid in channels_per_clusters.keys():
            if channel_nr in channels_per_clusters[uid]:
                uids_to_plot.append(uid)

        ui = 0
        for uid in uids_to_plot:
            spiketimes = data_io.spiketimes[recname][uid]
            idx_spiketimes = np.argwhere((spiketimes >= binfo.burst_onset + t0) & (spiketimes <= binfo.burst_onset + t1))
            spike_times = spiketimes[idx_spiketimes] - binfo.burst_onset

            y_min = burst_data.min()
            y_max = burst_data.max()

            sp_x, sp_y = [], []
            for s in spike_times:
                sp_x.extend([s[0], s[0], None])
                sp_y.extend([y_min, y_max, None])

            clr = clrs.random_color(ui % 10)
            ui += 1

            fig.add_scatter(
                name=uid,
                x=sp_x, y=sp_y,
                mode='lines', line=dict(color=clr, width=0.4),
                showlegend=True,
            )

        fig.update_xaxes(
            tickvals=np.arange(-500, 500, 100),
            title_text='time [ms]',
        )
        fig.update_yaxes(
            range=[y_min, y_max],
        )

        sname = figure_dir / session_id / 'raw_traces' / f'{channel_nr:.0f}' / bid
        utils.save_fig(fig, sname, display=False)

        n += 1
        if n >= n_to_print:
            break


#%% Plot raw channels per cluster

t0 = -200
t1 = 300
bursts = data_io.burst_df.query('recording_name == "241024_A_1_noblocker" and laser_level == 85')

print(bursts.index)

#%%
# bursts = data_io.burst_df.query('laser_level == 85')

for bid, binfo in bursts.iterrows():
    recname = binfo.recording_name

    input_file = filepaths.raw_dir / f'{recname}.raw'

    m = np.memmap(input_file.as_posix(), dtype=data_type)
    max_channels = 5
    for cluster_id, cinfo in data_io.cluster_df.iterrows():
        channels = channels_per_clusters[cluster_id]

        n_rows = len(channels)

        if n_rows > max_channels:
            channels = channels[:max_channels]
            n_rows = max_channels

        height = 0.2 * n_rows
        y_offset = 0.05
        y_spacing = 0.01
        y_height = (1 - 2*y_offset - (n_rows-1) * y_spacing ) / n_rows
        y_domains = {}
        x_domains = {}
        s_titles = {}

        for i in range(n_rows):
            y0 = 1 - y_offset - y_height * i - y_spacing * (i-1)
            y_domains[i+1] = [[y0-y_height, y0]]

            x_domains[i+1] = [[0.1, 0.9]]
            s_titles[i+1] = [f'{channels[i]:.0f}']

        fig = utils.make_figure(
            width=1, height=height,
            x_domains=x_domains, y_domains=y_domains,
            subplot_titles=s_titles,
        )

        spiketimes = data_io.spiketimes[binfo.recording_name][cluster_id]
        idx_spiketimes = np.argwhere((spiketimes >= binfo.burst_onset + t0) & (spiketimes <= binfo.burst_onset + t1))
        spike_times = spiketimes[idx_spiketimes] - binfo.burst_onset

        for chi, ch in enumerate(channels):
            pos = dict(row=chi+1, col=1)
            channel_index = np.arange(int(ch), m.size, data_nb_channels, dtype=int)
            channel_time = np.arange(0, channel_index.size, 1) / 20

            idx = np.argwhere((channel_time >= binfo.burst_onset + t0) & (channel_time <= binfo.burst_onset + t1))

            burst_data = m[channel_index[idx.flatten()]]

            burst_time = channel_time[idx.flatten()] - binfo.burst_onset

            burst_data = highpass_filter_convolution(burst_time / 1000, burst_data, 20)
            idx_cut = 500
            burst_time = burst_time[idx_cut:-idx_cut]
            burst_data = burst_data[idx_cut:-idx_cut]

            fig.add_scatter(
                x=burst_time, y=burst_data,
                mode='lines', line=dict(color='black', width=0.4),
                showlegend=False, **pos,
            )

            y_min = np.min(burst_data)
            y_max = np.max(burst_data)

            sp_x, sp_y = [], []
            for s in spike_times:
                sp_x.extend([s[0], s[0], None])
                sp_y.extend([y_min, y_max, None])

            fig.add_scatter(
                x=sp_x, y=sp_y,
                mode='lines', line=dict(color='green', width=0.4),
                showlegend=False,
                **pos,
            )

            fig.add_scatter(
                x=[0, 0], y=[y_min, y_max],
                mode='lines', line=dict(color='red', width=1, dash='2px'),
                showlegend=False,
                **pos,
            )
            fig.update_yaxes(
                range=[y_min, y_max],
                **pos,
            )

            fig.update_xaxes(
                tickvals=np.arange(-400, 400, 100) if pos['row'] == n_rows else [],
                **pos,
            )

        sname = figure_dir / session_id / 'raw_traces_per_cluster' / cluster_id / bid
        utils.save_fig(fig, sname, display=False)





