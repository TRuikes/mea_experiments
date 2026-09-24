from tqdm import tqdm
import numpy as np
from sonogenetics.analysis.lib.data_io import DataIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sonogenetics.analysis.lib.analysis_params import figure_dir_analysis
from sonogenetics.analysis.lib.sta_fitting import double_gaussian_fit
from matplotlib.gridspec import GridSpec
from scipy.ndimage import convolve
from scipy.signal import convolve as convolve_sig
import utils
import plotly.graph_objects as go
from sonogenetics.analysis.lib.sta_fitting import gaussian2D

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

    # Compute sta per cluster
    tasks = []
    for cid in tqdm(data_io.cluster_ids, desc='computing stas'):
        tasks.append({
            'recording_id': rec_id,
            'cluster_id': cid,
            'spike_train': data_io.spiketimes[rec_id][cid],
            'session_id': data_io.session_id,
            'checkerboard_params': checkerboard_params,
            'nb_repeats': nb_repeats,
            'non_repeated_sequence_portion': non_repeated_sequence_portion,
            'check_duration': check_duration,
            'check_onsets': check_onsets,
            'checkerboard': checkerboard,
            'chirp_temporal_dim': chirp_temporal_dim,
        })

    all_sta_output = utils.run_job(
        get_sta_single_cell,
        tasks=tasks,
        num_threads=10,
        debug=False,
    )
    return all_sta_output



def get_sta_single_cell(recording_id, cluster_id, spike_train, session_id, checkerboard_params, nb_repeats,
                        non_repeated_sequence_portion, check_duration, check_onsets, checkerboard,
                        chirp_temporal_dim):
        # Find spike-triggered-average of stimulus
        # spike_train = data_io.spiketimes[rec_id][cid]  # spiketrain in [ms]
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

        if len(sta) > 0:

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

        else:
            sta = None

        sta_output =  dict(
            psth=spike_counts,
            sta=sta,
            temporal_dim=chirp_temporal_dim,
            cluster_id=cluster_id,
            session_id=session_id,
            recording_id=recording_id,
        )

        sta_output = detect_receptive_field(
            sta_output=sta_output,
        )

        return sta_output


def detect_receptive_field(sta_output, smooth_alpha=0.8, max_time_window=15,):

    sta = sta_output['sta']
    if sta is None:
        sta_output['ellipse_coordinates'] = None
        sta_output['spatial_sta'] = None
        sta_output['spatial_sta_mask'] = None
        sta_output['temporal_sta_coords'] = None
        sta_output['temporal_sta'] = None
        sta_output['cell_delay'] = None
        return sta_output

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

    sta_output['ellipse_coordinates'] = ellipse_params
    sta_output['spatial_sta'] = spatial_sta
    sta_output['spatial_sta_mask'] = spatial_mask
    sta_output['temporal_sta_coords'] = cxy
    sta_output['temporal_sta'] = temporal_sta
    sta_output['cell_delay'] = cell_delay

    return sta_output



def plot_full_sta_single_cell(sta_output):

        sta = sta_output['sta']
        if sta is None:
            return

        chirp_temporal_dim = sta_output['temporal_dim']
        sid = sta_output['session_id']
        cid = sta_output['cluster_id']
        rid = sta_output['recording_id']

        vrange = np.max(np.abs(sta))
        vmin, vmax = -1 * vrange, vrange
        n_frames_to_show = chirp_temporal_dim
        ncols = 10
        nrows = int(np.ceil(n_frames_to_show / ncols))


        x_domains = {}
        y_domains = {}
        x_spacing = 0.01
        x_offset = 0.05
        y_spacing = 0.05
        y_offset = 0.05

        x_width = (1 - 2 * x_offset - (ncols-1) * x_spacing) / ncols
        y_height = (1 - 2 * y_offset - (nrows-1) * y_spacing) / nrows

        subplot_titles = {}
        for i in range(nrows):
            subplot_titles[i + 1] = []
            y1 = 1 - y_offset - (y_height+y_spacing) * i
            y0 = y1 - y_height
            y_domains[i+1] = [[y0, y1] for _ in range(ncols)]
            x_domains[i+1] = []
            for j in range(ncols):
                x0 = x_offset + (x_width+x_spacing) * j
                x1 = x0 + x_width
                x_domains[i+1].append([x0, x1])
                subplot_titles[i+1].append(f't: {i*ncols+j}')


        fig = utils.make_figure(
            width=1, height=1,
            x_domains=x_domains, y_domains=y_domains,
            equal_width_height='x',
            subplot_titles=subplot_titles,
            subplot_title_fontsize=4,
            subplot_title_xshift=0,
        )

        col = 0
        row = 0
        for i in range(n_frames_to_show):
            pos = dict(row=row+1, col=col+1)
            fig.add_trace(go.Heatmap(z=sta[i], showlegend=False, colorscale='RdBu_r',
                                     showscale=False, zmin=vmin, zmax=vmax), **pos)

            col += 1
            if col == ncols:
                col = 0
                row += 1


        savename = figure_dir_analysis / f'{sid}' / f'{rid}' / 'sta' / f'{cid}.png'
        # print(f'saving {savename}')
        if not savename.parent.exists():
            savename.parent.mkdir(parents=True)

        utils.save_fig(fig, savename, display=False)



def plot_sta_all_cells_with_rf(sta_outputs, n_sigma=2):

    level_factor = np.exp(-(n_sigma**2) / 2)

    ncols = 10
    nrows = int(np.ceil(len(sta_outputs) / ncols))

    nrows_max = 4
    if nrows > nrows_max:
        nrows = nrows_max

    x_domains = {}
    y_domains = {}
    x_spacing = 0.01
    x_offset = 0.01
    y_spacing = 0.01
    y_offset = 0.05

    x_width = (1 - 2 * x_offset - (ncols - 1) * x_spacing) / ncols
    y_height = (1 - 2 * y_offset - (nrows - 1) * y_spacing) / nrows

    subplot_titles = {}
    for i in range(nrows):
        subplot_titles[i + 1] = []
        y1 = 1 - y_offset - (y_height + y_spacing) * i
        y0 = y1 - y_height
        y_domains[i + 1] = [[y0, y1] for _ in range(ncols)]
        x_domains[i + 1] = []
        for j in range(ncols):
            x0 = x_offset + (x_width + x_spacing) * j
            x1 = x0 + x_width
            x_domains[i + 1].append([x0, x1])
            subplot_titles[i + 1].append(f't: {i * ncols + j}')

    try:
        fig = utils.make_figure(
            width=1, height=1,
            x_domains=x_domains, y_domains=y_domains,
            equal_width_height='x',
            subplot_title_fontsize=2,
        )
    except:
        fig = utils.make_figure(
            width=1, height=1,
            x_domains=x_domains, y_domains=y_domains,
            equal_width_height='y',
            subplot_title_fontsize=2,
        )

    col = 0
    row = 0
    page_i = 0
    subplot_titles = {}
    for so in sta_outputs:
        sid = so['session_id']
        cid = so['cluster_id']
        rid = so['recording_id']

        if so['sta'] is None:
            col += 1
            if col == ncols:
                col = 0
                row += 1
            continue

        if row >= nrows_max:

            utils.update_subplot_titles(fig, x_domains, y_domains, subplot_titles, fontsize=4, y_shift=0.001)
            savename = figure_dir_analysis / f'{sid}' / f'{rid}' / f'receptive_fields-{page_i}.png'
            utils.save_fig(fig, savename, display=False)


            try:
                fig = utils.make_figure(
                    width=1, height=1,
                    x_domains=x_domains, y_domains=y_domains,
                    equal_width_height='x',
                    subplot_title_fontsize=2,
                )
            except:
                fig = utils.make_figure(
                    width=1, height=1,
                    x_domains=x_domains, y_domains=y_domains,
                    equal_width_height='y',
                    subplot_title_fontsize=2,
                )

            subplot_titles = {}
            row = 0
            col = 0
            page_i += 1

        # Set position in figure
        pos = dict(row=row + 1, col=col + 1)


        # Extract cell parameters

        spatial_sta = so['spatial_sta']
        temporal_sta_coords = so['temporal_sta_coords']
        ellipse_coordinates = so['ellipse_coordinates']

        if ellipse_coordinates is None:
            col += 1
            if col == ncols:
                col = 0
                row += 1
            continue

        gaussian = gaussian2D(spatial_sta.shape, *ellipse_coordinates)
        amp, x0, y0, sigma_x, sigma_y, rot_angle = ellipse_coordinates
        vrange = np.max(np.abs(spatial_sta))


        gaussian_norm = gaussian / gaussian.sum()  # normalize gaussian weights


        signal = np.abs(np.sum(spatial_sta * gaussian_norm))
        residual = spatial_sta - (signal * gaussian_norm)
        noise = np.sqrt(np.mean(residual**2))  # RMS of residual

        SNR = signal / noise if noise != 0 else np.inf

        subplot_titles[(row+1, col+1)] = f'SNR: {SNR:.2f}'

        # Plot spatial sta
        print(pos)
        fig.add_trace(go.Heatmap(z=spatial_sta, showlegend=False, colorscale='RdBu_r',
                                 showscale=False, zmin=-vrange, zmax=vrange), **pos)

        # Plot RF contour
        if amp != 0:
            z = np.abs(gaussian)
            level = level_factor * np.max(z)

            fig.add_trace(go.Contour(
                z=z,
                contours=dict(
                    start=level,
                    end=level,
                    size=1,
                    coloring='lines',
                ),
                line=dict(
                    width=1,
                    dash='solid',
                ),
                colorscale=[[0, 'purple'], [1, 'purple']],            # opacity=alpha,
                showscale=False,
                showlegend=False,
            ), **pos)

            fig.add_scatter(
                x=[x0], y=[y0],
                mode='markers',
                marker=dict(color='purple', size=1, symbol='x'),
                showlegend=False,
                **pos,
            )

        col += 1
        if col == ncols:
            col = 0
            row += 1




    utils.update_subplot_titles(fig, x_domains, y_domains, subplot_titles, fontsize=4, y_shift=0.001)
    savename = figure_dir_analysis / f'{sid}' / f'{rid}' / f'receptive_fields-{page_i}.png'
    utils.save_fig(fig, savename, display=False)





