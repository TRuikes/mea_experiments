from tqdm import tqdm
import numpy as np
from typing import List, Tuple, cast, Any, Dict, Union
from pathlib import Path

import utils
from sonogenetics.analysis.lib.analysis_params import dataset_dir
from sonogenetics.analysis.data_list import data_list
from sonogenetics.analysis.lib.data_io import DataIO
import math
import matplotlib.pyplot as plt
from sonogenetics.analysis.lib.analysis_params import figure_dir_analysis
from utils import load_obj, make_figure, run_job, update_subplot_titles, save_fig
from pathlib import Path
from matplotlib.gridspec import GridSpec


CHECKERBOARD_FILE = Path(r'C:\thijs\sono_data\dmd_stimfiles\checkerboard.npy')
BINSOURCE_FILE = Path(r'C:\thijs\sono_data\dmd_stimfiles\binarysource1000Mbits')

chirp_temporal_dim = 40

checkerboard_params = {
    # 20 pxl / square
    'nb_checks_x': 30,
    'nb_checks_y': 30,
    'stimulus_frequency': 30,
    'nb_frames_by_sequence': 1200,  # Number of frames in each checkerboard sequence
}

repeated_sequence_portion = (
    0.5,
    1,
)  # portion holding the repeated sequence (second half)
non_repeated_sequence_portion = (
    0,
    0.5,
)  # portion holding the random sequence (first half)



def main():
    data_io = DataIO(dataset_dir)


    for sid in data_list:

        data_io.load_session(sid, load_pickle=False)


        # checkerboard_recordings = [r for r in data_io.recording_ids if 'checkerboard' in r]
        # for rec in checkerboard_recordings:
        #     plot_checkerboard_responses(data_io, rec)

        chirp_recordings = [r for r in data_io.recording_ids if 'chirp' in r]
        for rec in chirp_recordings:
            checkerboard = checkerboard_from_binary(
                data_io=data_io,
                rec_id=rec,
                binary_source_path=BINSOURCE_FILE
            )

            do_sta(data_io, rec, checkerboard)


def checkerboard_from_binary(
        data_io,
        rec_id,
        binary_source_path,
):
    # Extract trial data
    tinfo = data_io.burst_df.query(f'recording_name == "{rec_id}"').iloc[0]

    # Figure out how many checkerboard repeats we have
    dmd_duration = tinfo.dmd_burst_offset - tinfo.dmd_burst_onset  # [ms]
    check_duration = (1 / checkerboard_params['stimulus_frequency']) * checkerboard_params[
        'nb_frames_by_sequence'] * 1e3  # [ms]
    nb_check_repeats = int(np.floor(dmd_duration / check_duration))
    nb_frames_recording = int(nb_check_repeats * int(checkerboard_params['nb_frames_by_sequence'] / 2))

    nb_checks_x = checkerboard_params['nb_checks_x']
    nb_checks_y = checkerboard_params['nb_checks_y']

    binary_source_file = open(binary_source_path, mode="rb")
    checkerboard = np.zeros((nb_frames_recording, nb_checks_x, nb_checks_y), dtype="uint8")

    for frame in tqdm(range(nb_frames_recording)):
        image = np.zeros((nb_checks_x, nb_checks_y), dtype=float)

        for row in range(nb_checks_x):
            for col in range(nb_checks_y):
                bit_nb = (nb_checks_x * nb_checks_y * frame) + (nb_checks_x * row) + col
                binary_source_file.seek(bit_nb // 8)
                byte = int.from_bytes(binary_source_file.read(1), byteorder="big")
                bit = (byte & (1 << (bit_nb % 8))) >> (bit_nb % 8)
                if bit == 0:
                    image[row, col] = 0.0
                elif bit == 1:
                    image[row, col] = 1.0
                else:
                    message = "Unexpected bit value: {}".format(bit)
                    raise ValueError(message)


        checkerboard[frame, :, :] = image
    return checkerboard


def do_sta(data_io: DataIO, rec_id: str, checkerboard: np.ndarray):

    # Extract trial data
    tinfo = data_io.burst_df.query(f'recording_name == "{rec_id}"').iloc[0]

    # Figure out how many checkerboard repeats we have
    dmd_duration = tinfo.dmd_burst_offset - tinfo.dmd_burst_onset  # [ms]
    check_duration = (1 / checkerboard_params['stimulus_frequency']) * checkerboard_params[
        'nb_frames_by_sequence'] * 1e3  # [ms]
    nb_repeats = int(np.floor(dmd_duration / check_duration))

    check_onsets = np.arange(tinfo.dmd_burst_onset, tinfo.dmd_burst_offset, check_duration)[:nb_repeats]

    #
    #   EXTRACT PSTH FOR ALL CELLS
    #
    # Per cluster
    chirp_psth = {}
    chirp_stas = {}
    for cid in data_io.cluster_ids:
        spike_train = data_io.spiketimes[rec_id][cid]  # spiketrain in [ms]

        # output placeholders
        resp = []
        spike_counts = np.zeros((nb_repeats, int(checkerboard_params['nb_frames_by_sequence'] / 2)))

        # Grab data per check onset
        for ch_i, ch_onset in enumerate(check_onsets):
            # Define interval to extract data from
            t0 = ch_onset + non_repeated_sequence_portion[0] * check_duration
            t1 = ch_onset + non_repeated_sequence_portion[1] * check_duration

            # Store spikes as spiketrain and PSTH
            idx = np.where((spike_train >= t0) & (spike_train < t1))[0]
            resp.append(spike_train[idx] - t0)
            spike_counts[ch_i, :] = np.histogram(
                resp[-1],
                bins=int(checkerboard_params['nb_frames_by_sequence'] / 2),
                range=(0, t1 - t0),
            )[0]

        chirp_psth[cid] = spike_counts

        sta = np.zeros_like(checkerboard[:chirp_temporal_dim], dtype="float64")

        for sequence in range(nb_repeats):
            for frame in range(
                    chirp_temporal_dim, int(checkerboard_params['nb_frames_by_sequence'] / 2)
            ):  # should be made more robust to extact portion is in extract sequence?
                sta_frame_start = (
                        sequence * int(checkerboard_params['nb_frames_by_sequence']  / 2) + frame - chirp_temporal_dim
                )
                sta_frame_end = sequence * int(checkerboard_params['nb_frames_by_sequence']  / 2) + frame
                weight = spike_counts[sequence, frame]
                sta += weight * checkerboard[sta_frame_start:sta_frame_end, :, :]

        if np.max(np.abs(sta)) > 0:
            sta = sta / np.sum(spike_counts)
            # Bring values between -1 and 1
            sta -= np.median(sta)
            sta /= np.max(np.abs(sta))

            chirp_stas[cid] = sta
        else:
            chirp_stas[cid] = None
            continue

        #
        #   PLOT STA
        #
        vrange = np.max(np.abs(sta))
        vmin, vmax = -1 * vrange, vrange
        n_frames_to_show = chirp_temporal_dim
        ncols = 10
        nrows = int(np.ceil(n_frames_to_show / ncols))
        fig = plt.figure(figsize=(ncols * 3, nrows * 3))
        gs = GridSpec(nrows, ncols, figure=fig)
        for i in range(n_frames_to_show):
            ax = fig.add_subplot(gs[i // ncols, i % ncols])
            ax.imshow(sta[i], vmin=vmin, vmax=vmax, cmap='RdBu_r')
            ax.set_title(f"f: {i}", fontsize=14)
            ax.set_axis_off()
        # fig.suptitle(
        #     f"Cell {cell_id} - 3D Spike-Triggered Average (shape: {sta.shape})",
        #     fontsize=fontsize,
        # )
        # Save figure output
        savename = figure_dir_analysis / 'checkerboard' / 'sta' / f'{data_io.session_id}' / f'{cid}.png'
        print(f'saving {savename}')
        if not savename.parent.exists():
            savename.parent.mkdir(parents=True)

        plt.savefig(savename, dpi=300)
        plt.close(fig)









def plot_checkerboard_responses(data_io, rec_id):
    # Extract trial data
    tinfo = data_io.burst_df.query(f'recording_name == "{rec_id}"').iloc[0]

    # Figure out how many checkerboard repeats we have
    dmd_duration = tinfo.dmd_burst_offset - tinfo.dmd_burst_onset  # [ms]
    check_duration = (1 / checkerboard_params['stimulus_frequency']) * checkerboard_params[
        'nb_frames_by_sequence'] * 1e3  # [ms]
    nb_repeats = int(np.floor(dmd_duration / check_duration))
    check_onsets = np.arange(tinfo.dmd_burst_onset, tinfo.dmd_burst_offset, check_duration)[:nb_repeats]

    # Extract the checkerboard spiketrains and PSTH's
    check_responses = {}
    check_psth = {}

    # Per cluster
    for cid in data_io.cluster_ids:
        spike_train = data_io.spiketimes[rec_id][cid]  # spiketrain in [ms]

        # output placeholders
        resp = []
        spike_counts = np.zeros((nb_repeats, int(checkerboard_params['nb_frames_by_sequence'] / 2)))

        # Grab data per check onset
        for ch_i, ch_onset in enumerate(check_onsets):
            # Define interval to extract data from
            t0 = ch_onset + repeated_sequence_portion[0] * check_duration
            t1 = ch_onset + repeated_sequence_portion[1] * check_duration

            # Store spikes as spiketrain and PSTH
            idx = np.where((spike_train >= t0) & (spike_train < t1))[0]
            resp.append(spike_train[idx] - t0)
            spike_counts[ch_i, :] = np.histogram(
                resp[-1],
                bins=int(checkerboard_params['nb_frames_by_sequence'] / 2),
                range=(0, t1 - t0),
            )[0]
        check_responses[cid] = resp
        check_psth[cid] = spike_counts

    # Compute correlation for all clusters
    cluster_correlations = []
    for cid in data_io.cluster_ids:
        r = _even_odd_corr(check_psth[cid])
        cluster_correlations.append((cid, r))

    # Robust sorting: treat NaN as -infinity so high positive r values stay at the top
    cluster_correlations.sort(
        key=lambda item: -np.inf if np.isnan(item[1]) else item[1],
        reverse=True
    )

    # Setup matplotlib subplot grid
    size = int(math.sqrt(len(data_io.cluster_ids))) + 1
    fig, axes = plt.subplots(size, size, figsize=(size * 2, size * 2), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i in tqdm(range(size ** 2), desc="Plotting rasters for all cells"):
        ax = axes_flat[i]

        if i >= len(cluster_correlations):
            ax.axis('off')  # Hide unused subplots
            continue

        cluster_id, pearson_r = cluster_correlations[i]

        # Build contiguous segment arrays with NaN breaks
        x_plot, y_plot = [], []
        for burst_i, sp in enumerate(check_responses[cluster_id]):
            if sp.size > 0:
                x_plot.append(np.vstack([sp, sp, np.full(sp.size, np.nan)]).T.flatten())
                y_plot.append(np.vstack([np.ones(sp.size) * burst_i,
                                         np.ones(sp.size) * burst_i + 1,
                                         np.full(sp.size, np.nan)]).T.flatten())

        if x_plot:
            x_plot = np.hstack(x_plot)
            y_plot = np.hstack(y_plot)
            ax.plot(x_plot, y_plot, color='black', linewidth=0.5)

        # Set title and formatting
        title_str = f"r=NaN" if np.isnan(pearson_r) else f"r={pearson_r:.3f}"
        ax.set_title(title_str, fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)

    plt.tight_layout()

    # Save figure output
    savename = figure_dir_analysis / 'checkerboard' / f'{data_io.session_id}-{rec_id}.png'
    print(f'saving {savename}')
    if not savename.parent.exists():
        savename.parent.mkdir(parents=True)

    plt.savefig(savename, dpi=300)
    plt.close(fig)

def _even_odd_corr(rep_psths):
    """Pearson correlation between the mean of even-indexed and odd-indexed PSTH rows."""
    even, odd = rep_psths[0::2], rep_psths[1::2]
    if len(even) == 0 or len(odd) == 0:
        return np.nan
    mean_even, mean_odd = even.mean(0), odd.mean(0)
    if np.std(mean_even) == 0 or np.std(mean_odd) == 0:
        return np.nan  # a flat half -> correlation undefined
    return float(np.corrcoef(mean_even, mean_odd)[0, 1])



if __name__ == '__main__':
    main()