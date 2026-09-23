from tqdm import tqdm
import numpy as np
from sonogenetics.analysis.lib.data_io import DataIO
import matplotlib.pyplot as plt
from sonogenetics.analysis.lib.analysis_params import figure_dir_analysis
from sonogenetics.analysis.lib.sta_fitting import double_gaussian_fit
from matplotlib.gridspec import GridSpec
from scipy.ndimage import convolve
from scipy.signal import convolve as convolve_sig

def get_checkerboard_sta(data_io: DataIO, rec_id: str, checkerboard: np.ndarray, checkerboard_params: dict,
           non_repeated_sequence_portion: tuple, chirp_temporal_dim: int):

    # Extract checkerboard parameters
    stimulus_frequency = checkerboard_params['stimulus_frequency']
    nb_frames_per_sequence = checkerboard_params['nb_frames_by_sequence']

    # Extract trial data and figure out how many times the checkerboard is repeated
    tinfo = data_io.burst_df.query(f'recording_name == "{rec_id}"').iloc[0]
    dmd_duration = tinfo.dmd_burst_offset - tinfo.dmd_burst_onset  # [ms]
    check_duration = (1 / stimulus_frequency) * nb_frames_per_sequence * 1e3  # [ms]
    nb_repeats = int(np.floor(dmd_duration / check_duration))

    # Define onsets of each checkerboard
    check_onsets = np.arange(tinfo.dmd_burst_onset, tinfo.dmd_burst_offset, check_duration)[:nb_repeats]

    # Create placeholders for checkerboard output
    all_sta_output = {}

    # Compute sta per cluster
    for cid in tqdm(data_io.cluster_ids, desc='computing stas'):

        # Find spike-triggered-average of stimulus
        spike_train = data_io.spiketimes[rec_id][cid]  # spiketrain in [ms]
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

        # Normalize sta responses
        if np.max(np.abs(sta)) > 0:
            sta = sta / np.sum(spike_counts)
            # Bring values between -1 and 1
            sta -= np.median(sta)
            sta /= np.max(np.abs(sta))

        else:
            sta = None

        all_sta_output[cid] = dict(
            psth=spike_counts,
            sta=sta,
            temporal_dim=chirp_temporal_dim,
            cid=cid,
            sid=data_io.session_id,
        )

    return all_sta_output


def detect_receptive_field(sta_output, smooth_alpha=0.8, max_time_window=15,):

    sta = sta_output['sta']

    # Define the 2D neighborhood kernel
    # Center gets (alpha + (1 - alpha)) = 1, neighbors get (1 - alpha)
    kernel = np.array(
        [
            [1 - smooth_alpha, 1 - smooth_alpha, 1 - smooth_alpha],
            [1 - smooth_alpha, smooth_alpha + (1 - smooth_alpha), 1 - smooth_alpha],
            [1 - smooth_alpha, 1 - smooth_alpha, 1 - smooth_alpha],
        ]
    )

    # Expand kernel dimensions to align with the 3D array shape (time, height, width)
    kernel_3d = kernel[np.newaxis, :, :]

    # Convolve along spatial axes only (mode='constant', cval=0 mirrors zero-padding)
    receptive_field = convolve(sta, kernel_3d, mode="constant", cval=0.0)

    # Find the peak coordinates within the time window
    search_window = receptive_field[-max_time_window:, :, :]
    best_local = np.unravel_index(np.argmax(np.abs(search_window)), search_window.shape)

    # Offset the local time index to get the global time index
    cell_delay = best_local[0] + max(sta.shape[0] - max_time_window, 0)
    cx, cy = best_local[1], best_local[2]
    cxy = (cx, cy)

    temporal_sta = sta[:, cx, cy]
    spatial_sta = sta[cell_delay, :, :]

    spatial_mask = np.zeros_like(spatial_sta)
    spatial_mask[:] = np.nan

    # Fitt
    smoothing_kernel = np.full((3, 3), 0.2 / 9)
    smoothing_kernel[1, 1] += 0.8

    # Why are we smoothing again
    processed_spatial_sta = convolve_sig(
        spatial_sta, smoothing_kernel, mode="same", method="direct"
    )  # same to keep the same shape, direct to avoid artifacts of fft convolution on small arrays

    # Apply exponential ( == signed power-law) compression and threshold small values
    exponent = 1.25
    noise_threshold = 0.2

    processed_spatial_sta = (
        np.sign(processed_spatial_sta) * np.abs(processed_spatial_sta) ** exponent
    )
    peak = np.max(np.abs(processed_spatial_sta)) * 2
    processed_spatial_sta[
        np.abs(processed_spatial_sta) < peak * noise_threshold**exponent
    ] = 0

    try:
        ellipse_params, _ = double_gaussian_fit(processed_spatial_sta)
    except:
        ellipse_params = None

    sta_output['ellipse_fit'] = ellipse_params



def plot_sta(sta_output):

        sta = sta_output['sta']
        if sta is None:
            return
        chirp_temporal_dim = sta_output['temporal_dim']
        sid = sta_output['sid']
        cid = sta_output['cid']

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

        savename = figure_dir_analysis / 'checkerboard' / 'sta' / f'{sid}' / f'{cid}.png'
        print(f'saving {savename}')
        if not savename.parent.exists():
            savename.parent.mkdir(parents=True)

        plt.savefig(savename, dpi=300)
        plt.close(fig)