from pv_chip_aarhus.preprocessing.params import (data_sample_rate, data_type, data_nb_channels,
                                         data_trigger_channels, data_voltage_resolution,
                                         data_trigger_thresholds)
from pv_chip_aarhus.preprocessing.lib.filepaths import FilePaths
import utils
from pathlib import Path


from tqdm import tqdm
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
                if f'_{nr:03d}_' in rec:
                    print(f'\tskipping recording {rec}')
                    SKIP_RECORDING = True

        if SKIP_RECORDING:
            continue

        print(f'\treading recording: {rec}')

        trigger_data[rec] = {}

        # Load the recording file
        if filepaths.local_raw_dir is not None:
            recname = filepaths.local_raw_dir / f'{rec}.raw'
        else:
            recname = filepaths.raw_dir / f'{rec}.raw'

        data = np.memmap(recname, dtype=data_type)
        n_samples = int(data.size / data_nb_channels)
        rec_duration = (n_samples / data_sample_rate) / 60  # [min]

        print(f'\treading data ({rec_duration:.0f} min)')

        trigger_types = []
        if 'PA' in rec or 'pa' in rec:
            trigger_types.append('laser')

        if 'DMD' in rec or 'light' in rec:
            trigger_types.append('dmd')

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
                    x = (np.arange(i0, i1, 1) / data_sample_rate)
                    fig = utils.simple_fig(width=1, height=1, n_rows=1, n_cols=1)
                    fig.add_scatter(x=x, y=chdata, mode='lines', line=dict(color='black', width=1),
                                    showlegend=False, row=1, col=1)
                    fig.add_scatter(x=[x[0], x[-1]], y=np.ones(2) * data_trigger_thresholds['laser'], mode='lines',
                                    line=dict(color='red', width=1),
                                    showlegend=False, row=1, col=1)

                    if idx.size > 0:
                        fig.add_scatter(
                            x=x[idx], y=chdata[idx], mode='markers', marker=dict(color='green', size=1),
                            showlegend=False, row=1, col=1,
                        )

                    xticks = np.arange(i0/data_sample_rate, i1/data_sample_rate, 2)
                    xticks = [f'{xx:.0f}' for xx in xticks]
                    fig.update_xaxes(tickvals=xticks, title_text=f'time [s]')
                    fig.update_yaxes(tickvals=np.arange(0, 500, 4500), title_text='voltage [mV]')
                    savename = filepaths.proc_pp_figure_output / 'triggers' / rec / trigger_type / f'{i}'
                    utils.save_fig(fig, savename, display=True, verbose=False)

            if trigger_type == 'laser':
                # Process laser trigger times
                dt = np.diff(trigger_high)  # time difference between triggers, in ms

                trial_onsets_idx = np.concatenate([np.array([0]), np.where(dt > 1500)[0] + 1])
                burst_onsets_idx = np.concatenate([np.array([0]), np.where(dt > 5)[0] + 1])
                burst_offsets_idx = np.concatenate([np.where(dt > 5)[0], np.array([-1])])
                train_onsets = trigger_high[trial_onsets_idx]
                burst_onsets = trigger_high[burst_onsets_idx]
                burst_offsets = trigger_high[burst_offsets_idx]

                trigger_data[rec][trigger_type] = dict(
                    train_onsets=train_onsets,
                    burst_onsets=burst_onsets,
                    burst_offsets=burst_offsets,
                )

            elif trigger_type == 'dmd':
                print(f'note: DMD trigger min duration is 5 ms')

                dt = np.diff(trigger_high)  # time difference between triggers, in ms

                trial_onsets_idx = np.concatenate([np.array([0]), np.where(dt > 4500)[0] + 1])
                burst_onsets_idx = np.concatenate([np.array([0]), np.where(dt > 5)[0] + 1])
                burst_offsets_idx = np.concatenate([np.where(dt > 5)[0], np.array([-1])])

                train_onsets = trigger_high[trial_onsets_idx]
                burst_onsets = trigger_high[burst_onsets_idx]
                burst_offsets = trigger_high[burst_offsets_idx]

                trigger_data[rec][trigger_type] = dict(
                    train_onsets=train_onsets,
                    burst_onsets=burst_onsets,
                    burst_offsets=burst_offsets,
                )

            else:
                raise ValueError(f'need to implement {trigger_type}')

    utils.store_nested_dict(filepaths.proc_pp_triggers, trigger_data)