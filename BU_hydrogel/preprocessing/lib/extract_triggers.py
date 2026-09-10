from BU_hydrogel.preprocessing.params import (data_sample_rate, data_type, data_nb_channels,
                                         data_trigger_channels, data_voltage_resolution,
                                         data_trigger_thresholds)
from BU_hydrogel.preprocessing.lib.filepaths import FilePaths
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def extract_triggers(filepaths: FilePaths, update=False, visualize_detection=False,
                     recording_numbers_to_skip=None):
    print('\nProcessing trigger data')

    if filepaths.proc_pp_triggers.exists() and not update:
        print(f'\ttriggers already extracted')
        return

    trigger_data = {}

    for rec in filepaths.recording_names:

        SKIP_RECORDING = False
        if recording_numbers_to_skip is not None:
            for nr in recording_numbers_to_skip:
                if f'_{nr:.0f}_' in rec:
                    print(f'\tskipping recording {rec}')
                    SKIP_RECORDING = True

        if SKIP_RECORDING:
            continue

        print(f'\n\n\treading recording: {rec}')

        trigger_data[rec] = {}

        # Load the recording file
        recname = filepaths.raw_dir / f'{rec}.raw'

        data = np.memmap(recname, dtype=data_type)
        n_samples = int(data.size / data_nb_channels)
        rec_duration = (n_samples / data_sample_rate) / 60  # [min]

        print(f'\treading data ({rec_duration:.0f} min)')

        trigger_types = ['laser', 'dmd']
        # if 'PA' in rec or 'pa' in rec:
        #     trigger_types.append('laser')
        #
        # if 'DMD' in rec or 'dmd' in rec:
        #     trigger_types.append('dmd')

        assert len(trigger_types) > 0, 'no trigger types found!'

        for trigger_type in trigger_types:

            trigger_channel = data_trigger_channels[trigger_type]

            print(f'\t\treading {trigger_type}')

            trigger_high = np.array([])

            # Define indices of current channel in data object
            channel_index = np.arange(trigger_channel - 1, data.size, data_nb_channels)

            # Load the data into memory in chunks
            chunksize_s = 10
            chunksize = chunksize_s * data_sample_rate
            n_chunks = int(np.ceil(channel_index.size / chunksize))

            print(f'\t\treading data in {n_chunks} chunks')

            if visualize_detection:
                print(f'\t\tsaving figures in {filepaths.proc_pp_figure_output}')

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
                    t = ((idx+i0) / data_sample_rate) * 1e3  # [ms]

                    if idx.size > 0:
                        trigger_high = np.concat([trigger_high, t])

                elif trigger_type == 'dmd':
                    idx = np.where(chdata > data_trigger_thresholds['dmd'])[0]
                    t = ((idx+i0) / data_sample_rate) * 1e3  # [ms]

                    if idx.size > 0:
                        trigger_high = np.concat([trigger_high, t])

                else:
                    raise ValueError('error!')

                if visualize_detection:

                    # Plot trigger onsets
                    x = np.arange(i0, i1, 1) / data_sample_rate
                    subsample_idx = np.arange(0, x.size, 5).astype(int)

                    # Create figure and axis
                    fig, ax = plt.subplots(figsize=(6, 4))  # Adjust figsize as needed

                    # Main channel data line
                    ax.plot(x[subsample_idx], chdata[subsample_idx], color='black', linewidth=1)

                    # Threshold line (spans full x-range)
                    ax.axhline(y=data_trigger_thresholds['laser'], color='red', linewidth=1)

                    # Trigger markers (if present)
                    if idx.size > 0:
                        ax.scatter(x[idx], chdata[idx], color='green', s=1)

                    # Labels and Ticks
                    ax.set_xlabel('time [s]')
                    ax.set_ylabel('voltage [mV]')

                    # X-ticks every 2 seconds
                    xticks = np.arange(i0 / data_sample_rate, i1 / data_sample_rate, 2)
                    ax.set_xticks(xticks)

                    # Y-ticks
                    ax.set_yticks(np.arange(0, 500, 4500))

                    # Ensure output directories exist and save figure
                    savename = filepaths.proc_pp_figure_output / 'triggers' / rec / trigger_type / f'{i}.png'
                    savename.parent.mkdir(parents=True, exist_ok=True)

                    plt.tight_layout()
                    plt.savefig(savename, dpi=300)
                    plt.close(fig)  # Close the figure to free up memory (equivalent to display=False)

            if trigger_high.size == 0:
                print(F'{rec} does not have {trigger_type}')
                continue

            # Process laser trigger times
            dt = np.diff(trigger_high)  # time difference between triggers, in ms

            trial_onsets_idx = np.concatenate([np.array([0]), np.where(dt > 1500)[0] + 1])
            burst_onsets_idx = np.concatenate([np.array([0]), np.where(dt > 5)[0] + 1])
            burst_offsets_idx = np.concatenate([np.where(dt > 5)[0], np.array([-1])])
            train_onsets = trigger_high[trial_onsets_idx]
            burst_onsets = trigger_high[burst_onsets_idx]
            burst_offsets = trigger_high[burst_offsets_idx]

            burst_durations = burst_offsets - burst_onsets

            if trigger_type == 'laser':
                # The new laser has a pulse high when connected to PC,
                # this pulse is about 900 ms. Since stimulation is < 100 ms,
                # we can use 200 ms as a filter
                idx = burst_durations < 200
                burst_onsets = burst_onsets[idx]
                burst_offsets = burst_offsets[idx]
                dt = np.diff(burst_onsets)
                train_onsets_idx = np.concatenate([np.array([0]), np.where(dt > 2000)[0] + 1])
                train_onsets = burst_onsets[train_onsets_idx]


            print(f'RESULTS EXTRACT TRIGGER')
            print(f'{rec} {trigger_type}')
            print(f'n train: {train_onsets.size}')
            print(f'n burst: {burst_onsets.size}\n\n')

            trigger_data[rec][trigger_type] = dict(
                train_onsets=train_onsets,
                burst_onsets=burst_onsets,
                burst_offsets=burst_offsets,
            )

    utils.store_nested_dict(filepaths.proc_pp_triggers, trigger_data)