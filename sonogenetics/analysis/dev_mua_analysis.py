FILENAME = r"E:\sono\2026-03-25 mouse c57 617 Mekano6 A\raw\rec_2_A_20260325_pa_intensity_test.raw"
FIGURE_SAVEDIR =  r'E:\sono\figures\2026-03-25 mouse c57 617 Mekano6 A\MUA'
PROBE_FILE = r"E:\sono\2026-03-25 mouse c57 617 Mekano6 A\raw\2026-03-25_MEA_position.csv"
PICKLE_FILE = r"E:\sono\dmp.pkl"
TRIGGER_TYPE = 'laser'

T_PRE = 10
T_POST = 100
PLOT_CHANNEL = 100

from sonogenetics.preprocessing.params import (data_sample_rate, data_type, data_nb_channels,
                                         data_trigger_channels, data_voltage_resolution,
                                         data_trigger_thresholds)
import utils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt, find_peaks
from pathlib import Path
import matplotlib
# matplotlib.use('Agg')  # <-- use non-interactive backend
import pandas as pd
import pickle


def get_channel_index(data_size, channel_i):
    return np.arange(channel_i - 1, data_size, data_nb_channels, dtype=int)

def read_triggers(data, trigger_type):
    trigger_channel = data_trigger_channels[trigger_type]
    channel_index = get_channel_index(data.size, trigger_channel)

    trigger_high = np.array([])

    # Load the data into memory in chunks
    chunksize_s = 10
    chunksize = chunksize_s * data_sample_rate
    n_chunks = int(np.ceil(channel_index.size / chunksize))

    for i in tqdm(range(n_chunks), desc=f'reading chunks'):
        i0 = int(i * chunksize)
        i1 = int(i0 + chunksize)
        if i1 > channel_index.size - 1:
            i1 = channel_index.size - 1

        # Read data
        chdata = data[channel_index[i0:i1]]

        # Convert data to voltage
        chdata = chdata.astype(float)
        chdata = chdata - np.iinfo('uint16').min + np.iinfo('int16').min
        chdata = chdata * data_voltage_resolution

        # Detect trigger onsets
        if trigger_type == 'laser':
            idx = np.where(chdata > data_trigger_thresholds['laser'])[0]
            t = ((idx + i0) / data_sample_rate) * 1e3  # [ms]

            if idx.size > 0:
                trigger_high = np.concat([trigger_high, t])

        elif trigger_type == 'dmd':
            idx = np.where(chdata > data_trigger_thresholds['dmd'])[0]
            t = ((idx + i0) / data_sample_rate) * 1e3  # [ms]

            if idx.size > 0:
                trigger_high = np.concat([trigger_high, t])

        else:
            raise ValueError('error!')

    # Process laser trigger times
    dt = np.diff(trigger_high)  # time difference between triggers, in ms

    trial_onsets_idx = np.concatenate([np.array([0]), np.where(dt > 1500)[0] + 1])
    burst_onsets_idx = np.concatenate([np.array([0]), np.where(dt > 5)[0] + 1])
    burst_offsets_idx = np.concatenate([np.where(dt > 5)[0], np.array([-1])])
    train_onsets = trigger_high[trial_onsets_idx]
    burst_onsets = trigger_high[burst_onsets_idx]
    burst_offsets = trigger_high[burst_offsets_idx]

    return {
        'burst_onsets': burst_onsets,
        'burst_offsets': burst_offsets,
        'train_onsets': train_onsets,
    }



def get_filtered_daa_and_spikes(data, channel_i, trigger_time, trigger_i, train_i):
    # trigger tmie should be in [ms]
    # --- Convert ms → samples ---
    trigger_sample = int(trigger_time * data_sample_rate)
    pre_samples = int((T_PRE / 1000) * data_sample_rate)
    post_samples = int((T_POST / 1000) * data_sample_rate)

    # --- Padding for filtering ---
    pad_ms = 50
    pad_samples = int((pad_ms / 1000) * data_sample_rate)

    i0 = max(0, trigger_sample - pre_samples - pad_samples)
    channel_index = get_channel_index(data.size, channel_i)
    i1 = min(len(channel_index), trigger_sample + post_samples + pad_samples)

    # --- Load only required data ---
    raw_segment = data[channel_index[i0:i1]]

    # --- Bandpass filter ---
    nyq = 0.5 * data_sample_rate


    b0 = 400
    b1 = 3000

    if channel_i not in [126, 127, 128]:
        b, a = butter(3, [b0 / nyq, b1 / nyq], btype='band')
        filtered = filtfilt(b, a, raw_segment)
    else:
        filtered = raw_segment - np.mean(raw_segment)

    # --- Remove padding ---
    trim_start = (trigger_sample - pre_samples) - i0
    trim_end = trim_start + pre_samples + post_samples
    segment = filtered[trim_start:trim_end]
    # segment = raw_segment[trim_start:trim_end]

    # --- Time axis (ms) ---
    time_ms = (np.arange(len(segment)) - pre_samples) / data_sample_rate * 1000

    # --- MAD-based spike detection ---
    # noise_level = np.median(np.abs(segment)) / 0.6745
    # threshold = 3 * noise_level
    # Assume segment and time_ms are defined
    if channel_i not in [126, 127]:
        threshold = 3 * np.std(segment)

        # Positive peaks
        pos_peaks, _ = find_peaks(segment, height=threshold)

        # Negative peaks
        neg_peaks, _ = find_peaks(-segment, height=threshold)
        all_peaks = np.sort(np.concatenate([pos_peaks, neg_peaks]))
        spike_times = time_ms[all_peaks]


    else:
        threshold = 0
        spike_times = []


    return {
        'channel_i': channel_i,
        'segment': -segment,
        'spike_times': spike_times,
        'threshold': threshold,
        'trigger_i': trigger_i,
    }



def interactive_probe_map(
    data_list,
    mea_position,
    trigger_i,
    figsize=(12,12),
    linewidth=0.1,
    color='black',
    y_range=(-150,150),
    grid_size=15,
    margin=0.02
):
    """
    Interactive probe layout: click a channel to pop up in readable size.
    Uses channel_data['channel_i'] to map to electrode positions.
    """

    fig = plt.figure(figsize=figsize)

    # Normalize positions
    x_um = mea_position['x'].values
    y_um = mea_position['y'].values
    x_norm = (x_um - x_um.min()) / (x_um.max() - x_um.min())
    y_norm = (y_um - y_um.min()) / (y_um.max() - y_um.min())

    ax_size = (1.0 - margin*(grid_size+1)) / grid_size
    axes_list = []

    # Map channel_i to data
    data_trigger = [d for d in data_list if d['trigger_i'] == trigger_i]
    channel_to_data = {ch['channel_i']: ch for ch in data_trigger}

    for ch_idx in range(len(mea_position)):
        if ch_idx not in channel_to_data:
            continue  # skip channels without data

        channel_data = channel_to_data[ch_idx]
        segment = channel_data['segment']
        spike_times = channel_data['spike_times']
        threshold = channel_data.get('threshold', None)

        xpos = x_norm[ch_idx]
        ypos = y_norm[ch_idx]

        ax_x = margin + xpos*(1.0 - ax_size - 2*margin)
        ax_y = margin + ypos*(1.0 - ax_size - 2*margin)
        ax = fig.add_axes([ax_x, ax_y, ax_size, ax_size])
        axes_list.append((ax, ch_idx))

        n_samples = len(segment)
        time_ms = np.linspace(-T_PRE, T_POST, n_samples)

        # Plot segment
        ax.plot(time_ms, segment, linewidth=linewidth, color=color)

        # Spikes
        for t in spike_times:
            ax.axvline(t, color='red', linewidth=0.8)

        # Trigger
        ax.axvline(0, linestyle='--', color='blue', linewidth=0.5)

        # Threshold
        if threshold is not None:
            ax.axhline(threshold, linestyle='--', color='orange', linewidth=0.3)

        if ch_idx not in [126, 127]:
            ax.set_ylim(y_range)

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Electrode layout spike segments (click a channel to enlarge)")

    # --- Click event ---
    def on_click(event):
        for ax, ch_idx in axes_list:
            if event.inaxes == ax:
                channel_data = channel_to_data[ch_idx]
                segment = channel_data['segment']
                spike_times = channel_data['spike_times']
                threshold = channel_data.get('threshold', None)

                fig2, ax2 = plt.subplots(figsize=(8,4))
                n_samples = len(segment)
                time_ms = np.linspace(-T_PRE, T_POST, n_samples)
                ax2.plot(time_ms, segment, linewidth=1, color=color)
                for t in spike_times:
                    ax2.axvline(t, color='red', linewidth=1)
                ax2.axvline(0, linestyle='--', color='blue', linewidth=1)
                if threshold is not None:
                    ax2.axhline(threshold, linestyle='--', color='orange', linewidth=1)

                if ch_idx not in [126, 127]:
                    ax2.set_ylim(y_range)
                ax2.set_xlabel("Time (ms)")
                ax2.set_ylabel("Amplitude")
                ax2.set_title(f"Channel {ch_idx}")
                ax2.grid(True)
                plt.show()
                break

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()


def interactive_raster_probe_map_trials(
    data_list,
    mea_position,
    figsize=(12,12),
    spike_color='black',
    trigger_color='red',
    grid_size=15,
    margin=0.02
):
    """
    Interactive probe layout raster plot:
    - Each channel shows a raster with rows = triggers (trials)
    - Click a channel to enlarge its raster
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    fig = plt.figure(figsize=figsize)

    # Normalize positions
    x_um = mea_position['x'].values
    y_um = mea_position['y'].values
    x_norm = (x_um - x_um.min()) / (x_um.max() - x_um.min())
    y_norm = (y_um - y_um.min()) / (y_um.max() - y_um.min())

    ax_size = (1.0 - margin*(grid_size+1)) / grid_size
    axes_list = []

    # --- Group data: channel -> list of trials ---
    channel_trials = defaultdict(list)
    for d in data_list:
        channel_trials[d['channel_i']].append(d)

    # # Sort trials per channel by trigger_i (important!)
    # for ch in channel_trials:
    #     channel_trials[ch] = sorted(channel_trials[ch], key=lambda x: x['trigger_i'])

    for ch_idx in range(len(mea_position)):
        if ch_idx not in channel_trials:
            continue

        trials = channel_trials[ch_idx]

        xpos = x_norm[ch_idx]
        ypos = y_norm[ch_idx]

        ax_x = margin + xpos*(1.0 - ax_size - 2*margin)
        ax_y = margin + ypos*(1.0 - ax_size - 2*margin)
        ax = fig.add_axes([ax_x, ax_y, ax_size, ax_size])
        axes_list.append((ax, ch_idx))

        # --- Plot raster: each trial = one row ---
        for i, trial in enumerate(trials):
            spike_times = trial['spike_times']
            for t in spike_times:
                ax.vlines(t, i, i+1, color=spike_color, linewidth=0.4)

        # Trigger line (same for all rows)
        ax.axvline(0, linestyle='--', color=trigger_color, linewidth=0.5)

        ax.set_ylim(0, len(trials))
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Electrode layout raster (rows = triggers, click to enlarge)")

    # --- Click event ---
    def on_click(event):
        for ax, ch_idx in axes_list:
            if event.inaxes == ax:
                trials = channel_trials[ch_idx]

                fig2, ax2 = plt.subplots(figsize=(8, 5))

                for i, trial in enumerate(trials):
                    spike_times = trial['spike_times']
                    for t in spike_times:
                        ax2.vlines(t, i, i+1, color=spike_color, linewidth=0.8)

                ax2.axvline(0, linestyle='--', color=trigger_color, linewidth=1)

                ax2.set_ylim(0, len(trials))
                ax2.set_xlabel("Time (ms)")
                ax2.set_ylabel("Trigger index (sorted)")
                ax2.set_title(f"Channel {ch_idx} Raster (multi-trial)")

                ax2.grid(True, axis='x', alpha=0.3)
                plt.show()
                break

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()


def main(filename, trigger_channel):

    # Load channel data
    # data = np.memmap(filename, dtype=data_type)
    # filename = Path(filename)
    # n_samples = int(data.size / data_nb_channels)
    # rec_duration = (n_samples / data_sample_rate) / 60  # [min]
    #
    #
    # # Define indices of current channel in data object
    # triggers = read_triggers(data, trigger_channel)
    # n_trains = len(triggers['train_onsets'])
    # n_bursts = len(triggers['burst_onsets'])
    #
    # # Print file information
    # print(f'\t\t\t{filename.name}\n')
    # print(f'\trecording duration: {rec_duration:.0f} min\n')
    # print(f'\tdetected {n_trains} trains on {trigger_channel} trigger')
    # print(f'\tdetected {n_bursts} bursts on {trigger_channel} trigger')
    #
    #
    # ### CUT DATA AROUND TRIGGER
    # trigger_time = triggers['burst_onsets'][0] / 1e3  # in [s]
    #
    # job_list = []
    #
    # # for train_i in range(n_trains-1):
    # train_i = 10
    # burst_onsets_idx = np.where(
    #     (triggers['burst_onsets'] >= triggers['train_onsets'][train_i]) &
    #     (triggers['burst_onsets'] < triggers['train_onsets'][train_i+1])
    # )[0]
    # print(burst_onsets_idx.size)

    for burst_i, burst_time in enumerate(triggers['burst_onsets'][burst_onsets_idx]):
        for i in range(0, data_nb_channels):
            job_list.append({
                'data': data,
                'channel_i': i,
                'trigger_time': trigger_time / 1e3,
                'trigger_i': burst_i,
                'train_i': train_i,
            })

    spike_data = utils.run_job(
        job_fn=get_filtered_daa_and_spikes,
        num_threads=12,
        tasks=job_list,
        debug=False,
    )

    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(spike_data, f)

    with open(PICKLE_FILE, "rb") as f:
        spike_data = pickle.load(f)

    mea_position = pd.read_csv(PROBE_FILE, index_col=0, header=0)

    interactive_raster_probe_map_trials(
        data_list=spike_data,
        mea_position=mea_position,
    )

    # interactive_probe_map(
    #     data_list=spike_data,
    #     mea_position=mea_position,
    #     trigger_i=2,
    # )

if __name__ == "__main__":
    main(FILENAME, TRIGGER_TYPE)