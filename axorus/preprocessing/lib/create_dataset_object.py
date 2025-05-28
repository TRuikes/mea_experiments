from axorus.preprocessing.lib.filepaths import FilePaths
import pandas as pd
import utils
import h5py


names_as_int = (
    'burst_count', 'burst_duration',
    'burst_period', 'duty_cycle', 'electrode',
    'laser_level',
    'laser_x', 'laser_y',
    'train_count',
    'train_period',
    'ch', 'depth', 'sh', 'n_spikes'
)


class Dataset:

    """"
    /
        /RECNR
            /SPIKES
                /UNIT ID -> np.array with spiketimes

            /TRIGGERS
                /LASER -> dict with per trigger time a dict containing its meta info

    """
    def __init__(self, filepaths: FilePaths):

        return


def create_dataset_object(filepaths: FilePaths, include_waveforms=True):
    print('\nCreating dataset object')
    train_df = pd.read_csv(filepaths.proc_pp_trials, index_col=0, header=0)
    spiketimes = utils.load_nested_dict(filepaths.proc_pp_spiketimes)

    if include_waveforms:
        waveforms = utils.load_nested_dict(filepaths.proc_pp_waveforms)

    triggers = utils.load_nested_dict(filepaths.proc_pp_triggers)
    cluster_info = pd.read_csv(filepaths.proc_pp_clusterinfo, index_col=0, header=0)
    mea_position = pd.read_csv(filepaths.raw_mea_position, index_col=0, header=0)

    n_trials_triggers = 0
    n_bursts_triggers = 0
    for rec_id in filepaths.recording_names:
        if 'dmd' in rec_id or 'checkerboard' in rec_id or 'chirp' in rec_id or 'wl' in rec_id:
            print(f'skipping {rec_id}, not adding DMD data')
            # n_trials_triggers += 1
            continue
        train_onsets = triggers[rec_id]['laser']['train_onsets']
        n_trials_triggers += len(train_onsets)
        print(f'{rec_id}: {len(train_onsets):.0f} trains')

        burst_onsets = triggers[rec_id]['laser']['burst_onsets']
        n_bursts_triggers += len(burst_onsets)

    # Verify that every burst registered in the trials dataframe
    # also exists in the trigger data
    assert n_trials_triggers == train_df.shape[0]
    assert n_bursts_triggers == train_df.train_count.sum()

    write_file = filepaths.dataset_file_waveforms if include_waveforms else filepaths.dataset_file

    with h5py.File(write_file, 'w') as f:

        for rec_id in filepaths.recording_names:
            print(f'\tloading {rec_id}')
            # grp = f.create_group(rec_id)

            train_rec_df = train_df.query('recording_name == @rec_id')

            if 'pa' in rec_id:
                assert train_rec_df.shape[0] > 0

            if 'pa' in rec_id or rec_id in ['241024_A_1_noblocker',
                                            '241024_A_2_noblocker']:
                # Store laser trigger data
                burst_offset = 0
                for train_id, trial_info in train_rec_df.iterrows():
                    train_onsets = triggers[rec_id]['laser']['train_onsets']
                    burst_onsets = triggers[rec_id]['laser']['burst_onsets']
                    burst_offsets = triggers[rec_id]['laser']['burst_offsets']

                    burst_count = int(trial_info.train_count)

                    for burst_i in range(burst_count):
                        burst_id = f'{train_id}-{burst_i:02d}'
                        trigger_i = int(burst_offset + burst_i)

                        burst = f.create_group(f'{rec_id}/laser/{burst_id}')

                        to_store = dict(
                            train_onset=train_onsets[int(trial_info.rec_train_i)],
                            burst_onset=burst_onsets[trigger_i],
                            burst_offset=burst_offsets[trigger_i],
                            **trial_info,
                        )

                        for k, v in to_store.items():
                            if isinstance(v, str):
                                burst.create_dataset(k, data=v, dtype=h5py.string_dtype(encoding='utf-8'))
                            elif k in names_as_int:
                                burst.create_dataset(k, data=int(v), dtype='int')
                            else:
                                burst.create_dataset(k, data=v, dtype='float')
                    burst_offset += burst_count

            elif 'dmd' in rec_id or 'checkerboard' in rec_id or 'chirp' in rec_id or 'wl' in rec_id:
                print(f'need to add dmd still...')
            else:
                raise ValueError(f'{rec_id}?')

            # Store spiketimes
            for cluster_id, cinfo in cluster_info.iterrows():

                cluster = f.create_group(f'{rec_id}/clusters/{cluster_id}')

                for k, v in cinfo.items():
                    if isinstance(v, str):
                        cluster.create_dataset(k, data=v, dtype=h5py.string_dtype(encoding='utf-8'))
                    elif k in names_as_int:
                        cluster.create_dataset(k, data=int(v), dtype='int')
                    else:
                        cluster.create_dataset(k, data=v, dtype='float')

                cluster.create_dataset('spiketimes', data=spiketimes[rec_id][cluster_id])

                ch = cinfo.ch
                cluster.create_dataset('cluster_x', data=int(mea_position.loc[ch+1].x))
                cluster.create_dataset('cluster_y', data=int(mea_position.loc[ch+1].y))

                if include_waveforms:
                    cluster.create_dataset('waveforms', data=waveforms[rec_id][cluster_id])

        if include_waveforms:
            print(f'saved datasetfile: {filepaths.dataset_file_waveforms}')
        else:
            print(f'saved datasetfile: {filepaths.dataset_file}')
