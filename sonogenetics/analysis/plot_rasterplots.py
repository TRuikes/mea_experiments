import sys
from pathlib import Path
import numpy as np
import utils
from sonogenetics.analysis.lib.analysis_params import dataset_dir, data_list
from sonogenetics.analysis.lib.data_io import DataIO
from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis


data_io = DataIO(dataset_dir)
# session_id = data_io.sessions[2]
session_id = data_list[0]

print(f'Loading data: {session_id}')
data_io.load_session(session_id, load_waveforms=False, load_pickle=True)  # type: ignore

rec_id = data_io.recording_ids[0]

train_df = data_io.train_df.query('rec_id == @rec_id')
print(train_df.shape, rec_id)

t_pre: int = 100
t_after: int = 300
stepsize: int = 5
binwidth: int = 20
bin_centres = np.arange(-t_pre, t_after, stepsize)
baseline = [-100, -0]
response_window = [0, 200]
n_bins: int = bin_centres.size


for cluster_id in data_io.cluster_ids:

    spiketrain: np.ndarray = data_io.spiketimes[rec_id][cluster_id]


    for ec, df in train_df.groupby('electrode'):

        # Setup figure layout
        fig = utils.make_figure(
            width=1,
            height=1.5,
            x_domains={
                1: [[0.2, 0.99]],
            },
            y_domains={
                1: [[0.1, 0.9]]
            },
        )

        # Setup variables for plotting
        burst_offset = 0
        x_plot, y_plot = [], []
        x_lines_laser, y_lines_laser = [], []
        x_lines_dmd, y_lines_dmd = [], []

        yticks = []
        ytext = []
        pos = dict(row=1, col=1)

        for tid, r in df.iterrows():
            burst_onsets: np.ndarray = data_io.burst_df.query(
                'train_id == @tid').laser_burst_onset.values  # type: ignore
            n_trains: int = len(burst_onsets)

            # Create placeholder for data
            binned_sp: np.ndarray = np.zeros((n_trains, n_bins), dtype=int)
            spike_times = []

            for burst_i, burst_onset in enumerate(burst_onsets):
                t0: float = burst_onset + bin_centres[0] - binwidth / 2
                t1: float = burst_onset + bin_centres[-1] + binwidth / 2
                idx: np.ndarray = np.where((spiketrain >= t0) & (spiketrain < t1))[0]

                # Append the spiketimes, relative to burst onset
                spike_times = spiketrain[idx] - burst_onset

                # for sp in spike_times:
                x_plot.append(np.vstack([spike_times, spike_times, np.full(spike_times.size, np.nan)]).T.flatten())
                y_plot.append(np.vstack([np.ones(spike_times.size) * burst_offset,
                                        np.ones(spike_times.size)* burst_offset +1, np.full(spike_times.size, np.nan)]).T.flatten())
                burst_offset += 1

        x_plot = np.hstack(x_plot)
        y_plot = np.hstack(y_plot)


        fig.add_scatter(
            x=x_plot[:], y=y_plot[:],
            mode='lines', line=dict(color='black', width=1),
            showlegend=False,
            **pos,
        )

        fig.update_xaxes(
            tickvals=np.arange(-500, 500, 100),
            title_text=f'time [ms]',
            range=[bin_centres[0] - 1, bin_centres[-1] + 1],
            **pos,
        )

        fig.update_yaxes(
            range=[0, burst_offset],
            tickvals=yticks,
            ticktext=ytext,
            **pos,
        )

        savename = (figure_dir_analysis / data_io.session_id /
                    'raster_plots_dev' / f'{cluster_id}_{ec}')

        utils.save_fig(fig, savename)




