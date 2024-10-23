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


def create_dataset_object(filepaths: FilePaths):
    train_df = pd.read_csv(filepaths.proc_pp_trials, index_col=0, header=0)
    spiketimes = utils.load_nested_dict(filepaths.proc_pp_spiketimes)
    triggers = utils.load_nested_dict(filepaths.proc_pp_triggers)
    cluster_info = pd.read_csv(filepaths.proc_pp_clusterinfo, index_col=0, header=0)
    mea_position = pd.read_csv(filepaths.raw_mea_position, index_col=0, header=0)

    rec_id = '241016_A_1_noblocker'
    stimes_rec = spiketimes[rec_id]
    laser_rec = triggers[rec_id]['laser']

    train_onsets = laser_rec['train_onsets']
    burst_onsets = laser_rec['burst_onsets']
    burst_offsets = laser_rec['burst_offsets']

    # Verify that every burst registered in the trials dataframe
    # also exists in the trigger data
    assert train_onsets.size == train_df.shape[0]
    n_trials = train_onsets.size

    assert len(train_df.train_count.unique()) == 1
    burst_count = train_df.iloc[0].train_count
    n_bursts = n_trials * burst_count

    assert burst_onsets.size == n_bursts

    with h5py.File(filepaths.dataset_file, 'w') as f:

        grp = f.create_group(rec_id)

        # Store laser trigger data
        for train_id, trial_info in train_df.iterrows():

            for burst_i in range(burst_count):
                burst_id = f'{train_id}-{burst_i}'
                trigger_i = int(trial_info.train_i * burst_count + burst_i)

                burst = grp.create_group(f'laser/{burst_id}')

                to_store = dict(
                    train_onset=train_onsets[int(trial_info.train_i)],
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

        # Store spiketimes
        for cluster_id, cluster_info in cluster_info.iterrows():

            cluster = grp.create_group(f'clusters/{cluster_id}')

            for k, v in cluster_info.items():
                if isinstance(v, str):
                    cluster.create_dataset(k, data=v, dtype=h5py.string_dtype(encoding='utf-8'))
                elif k in names_as_int:
                    cluster.create_dataset(k, data=int(v), dtype='int')
                else:
                    cluster.create_dataset(k, data=v, dtype='float')

            cluster.create_dataset('spiketimes', data=spiketimes[rec_id][cluster_id])

            ch = cluster_info.ch
            cluster.create_dataset('cluster_x', data=int(mea_position.loc[ch+1].x))
            cluster.create_dataset('cluster_y', data=int(mea_position.loc[ch+1].y))

        print(f'saved datasetfile: {filepaths.dataset_file}')