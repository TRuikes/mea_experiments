from sonogenetics.preprocessing.lib.filepaths import FilePaths
from sonogenetics.preprocessing.params import manuall_edited_sessions
import pandas as pd
import utils
import h5py
import numpy as np

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


def create_dataset_object(filepaths: FilePaths, include_waveforms=True,
                          recording_numbers_to_skip=None):
    print('\nCreating dataset object')
    train_df = pd.read_csv(filepaths.proc_pp_trials, index_col=0, header=0)
        
    spiketimes = utils.load_nested_dict(filepaths.proc_pp_spiketimes)

    if include_waveforms:
        waveforms = utils.load_nested_dict(filepaths.proc_pp_waveforms)

    triggers = utils.load_nested_dict(filepaths.proc_pp_triggers)
    cluster_info = pd.read_csv(filepaths.proc_pp_clusterinfo, index_col=0, header=0)
    mea_position = pd.read_csv(filepaths.mea_position_file, index_col=0, header=0)

        # Add cluster x and y to data
    for i, r in cluster_info.iterrows():
        cluster_info.at[i, 'cluster_x'] = mea_position.loc[r.ch+1].x
        cluster_info.at[i, 'cluster_y'] = mea_position.loc[r.ch+1].y
        

    dmd_n_trials_triggers = 0
    dmd_n_bursts_triggers = 0
    pa_n_trials_triggers = 0
    pa_n_bursts_triggers = 0
    pa_recordings_included = []
    dmd_recordings_included = []

    for rec_id in filepaths.recording_names:

                        
        SKIP_RECORDING = False
        if recording_numbers_to_skip is not None:
            for nr in recording_numbers_to_skip:
                if f'_{nr:.0f}_' in rec_id:
                    print(f'\tskipping recording {rec_id}')
                    SKIP_RECORDING = True

        if SKIP_RECORDING:
            continue
        
        print(f'{rec_id}')
        has_triggers = False
        has_pa = False
        has_dmd = False

        if 'PA' in rec_id or 'buSTIM1' in rec_id or 'pilot_021126' in rec_id or 'pilot021626' in rec_id or 'pa' in rec_id:
            pa_recordings_included.append(rec_id)

            pa_train_onsets = triggers[rec_id]['laser']['train_onsets']
            print(f'{rec_id} - len train onsets: {len(pa_train_onsets)}')
            pa_n_trials_triggers += len(pa_train_onsets)

            pa_burst_onsets = triggers[rec_id]['laser']['burst_onsets']
            pa_n_bursts_triggers += len(pa_burst_onsets)

            print(f'\tAdding PA triggers')
            has_triggers = True
            has_pa = True

        if 'DMD' in rec_id or 'chirp' in rec_id or 'dmd' in rec_id:
            dmd_recordings_included.append(rec_id)

            dmd_train_onsets = triggers[rec_id]['dmd']['train_onsets']
            dmd_n_trials_triggers += len(dmd_train_onsets)

            dmd_burst_onsets = triggers[rec_id]['dmd']['burst_onsets']
            dmd_n_bursts_triggers += len(dmd_burst_onsets)

            print(f'\tAdding DMD triggers')
            has_triggers = True
            has_dmd = True


    assert has_triggers

    # Check PA triggers
    print(pa_recordings_included)
    train_df_check = train_df.query('recording_name in @pa_recordings_included')

    # Verify that every burst registered in the trials dataframe
    # also exists in the trigger data
    if filepaths.sid not in manuall_edited_sessions:

        if has_pa:
            n_pa_trains = train_df_check.has_laser.sum()
            assert pa_n_trials_triggers == n_pa_trains, f'{filepaths.sid}, {pa_n_trials_triggers} - {n_pa_trains}'
            n_bursts_train_df = train_df_check.laser_burst_count.sum()
            assert pa_n_bursts_triggers == n_bursts_train_df, f'{filepaths.sid}, {pa_n_bursts_triggers}, {n_bursts_train_df}'

        if has_dmd:
            # Check DMD triggers
            train_df_check = train_df.query('recording_name in @dmd_recordings_included')
            assert dmd_n_trials_triggers == train_df_check.shape[0]
            assert dmd_n_bursts_triggers == train_df_check.dmd_burst_count.sum()

    write_file = filepaths.dataset_file_waveforms if include_waveforms else filepaths.dataset_file

    if not write_file.parent.exists():
        write_file.parent.mkdir(parents=True)        

    with h5py.File(write_file, "w") as f:
    
        # -----------------------------
        # 0) Top-level cluster info table
        # -----------------------------
        # Determine max length of index if it's string/object
        if cluster_info.index.dtype.kind in "O" or cluster_info.index.dtype.kind in "U" or cluster_info.index.dtype.kind in "S":
            maxlen_index = cluster_info.index.astype(str).map(len).max()
            index_dtype = f"S{maxlen_index}"
        else:  # numeric index
            index_dtype = "i8"

        # Add index as first field
        cluster_dtype_fields = [("index", index_dtype)]
        for col in cluster_info.columns:
            if cluster_info[col].dtype.kind in "i":
                cluster_dtype_fields.append((col, "i4"))
            elif cluster_info[col].dtype.kind in "f":
                cluster_dtype_fields.append((col, "f4"))
            else:
                maxlen_col = cluster_info[col].astype(str).map(len).max()
                cluster_dtype_fields.append((col, f"S{maxlen_col}"))

        # Create structured array
        cluster_table = np.zeros(len(cluster_info), dtype=np.dtype(cluster_dtype_fields))

        for i, (idx, row) in enumerate(cluster_info.iterrows()):
            # Store original index
            if isinstance(idx, str):
                cluster_table[i]["index"] = idx.encode("utf-8")
            else:
                cluster_table[i]["index"] = idx

            for col in cluster_info.columns:
                val = row[col]
                if isinstance(val, str):
                    cluster_table[i][col] = val.encode("utf-8")
                elif pd.isna(val):
                    cluster_table[i][col] = np.nan if cluster_info[col].dtype.kind in "f" else -1
                else:
                    cluster_table[i][col] = val

        # Save dataset
        f.create_dataset("clusters/metadata", data=cluster_table,
                        compression="gzip", chunks=True)
        
        # Patches for broken data
        if filepaths.sid in manuall_edited_sessions:
            if filepaths.sid == '2025-12-17 rat P23H 3153 A':
                train_df = train_df.loc[train_df.index < 'tid_2025-12-17 rat P23H 3153 A_038']
                print(f'{filepaths.sid}: cutting rows from trial data')



        # -----------------------------
        # 1) Per recording data
        # -----------------------------
        for rec_id in filepaths.recording_names:

            # Skip recordings if needed
            if recording_numbers_to_skip and any(f'{nr:03d}' in rec_id for nr in recording_numbers_to_skip):
                print(f"Skipping recording {rec_id}")
                continue

            print(f"Loading {rec_id}")
            train_rec_df = train_df.query("recording_name == @rec_id").copy()
            if train_rec_df.empty:
                continue

            stim_type = 'laser'
            # # Determine stim type
            # if "_PA_" in rec_id:
            #     stim_type = "laser"
            # elif "_DMD_" in rec_id:
            #     stim_type = "dmd"
            # elif "_PADMD_" in rec_id:
            #     stim_type = "padmd"
            # else:
            #     raise ValueError(f"Unknown recording type: {rec_id}")

            train_rec_df['stimtype'] = stim_type
            
            # Group for this recording
            rec_grp = f.require_group(f"recordings/{rec_id}")

            # -----------------------------
            # 1a) Triggers array
            # -----------------------------
            bursts_list = []
            burst_offset = 0

            # Ticker to index into laser and dmd trigger onsets
            # This becomes relevent if in 1 recording there are trials with
            # andwithout dual stimulation. In which case there are more
            # dmd or laser triggers
            # The ticks are a bit redundant, but the could would break if there
            # are fewer detected triggers than trials, so its a nice backup
            dmd_tick, laser_tick = 0, 0
            dmd_burst_tick, laser_burst_tick = 0, 0

            for train_id, trial_info in train_rec_df.iterrows():
                if trial_info['has_dmd']:
                    burst_count = int(trial_info['dmd_burst_count'])
                elif trial_info['has_laser']:
                    burst_count = int(trial_info['laser_burst_count'])
                else:
                    raise ValueError(f"laser or dmd should have bursts")

                if trial_info['has_laser']:
                    laser_burst_count = int(trial_info.laser_burst_count)
                    laser_train_onsets = triggers[rec_id]["laser"]["train_onsets"]
                    laser_burst_onsets = triggers[rec_id]["laser"]["burst_onsets"]
                    laser_burst_offsets = triggers[rec_id]["laser"]["burst_offsets"]
                else:
                    laser_train_onsets = None
                    laser_burst_onsets = None
                    laser_burst_offsets = None

                if trial_info['has_dmd']:
                    dmd_train_onsets = triggers[rec_id]["dmd"]["train_onsets"]
                    dmd_burst_onsets = triggers[rec_id]["dmd"]["burst_onsets"]
                    dmd_burst_offsets = triggers[rec_id]["dmd"]["burst_offsets"]
                else:
                    dmd_train_onsets = None
                    dmd_burst_onsets = None
                    dmd_burst_offsets = None

                for burst_i in range(burst_count):

                    bursts_list.append([
                        dmd_train_onsets[dmd_tick] if trial_info['has_dmd'] else 0,
                        dmd_burst_onsets[dmd_burst_tick]  if trial_info['has_dmd'] else 0,
                        dmd_burst_offsets[dmd_burst_tick] if trial_info['has_dmd'] else 0,
                        laser_train_onsets[laser_tick] if trial_info['has_laser'] else 0,
                        laser_burst_onsets[laser_burst_tick] if trial_info['has_laser'] else 0,
                        laser_burst_offsets[laser_burst_tick] if trial_info['has_laser'] else 0,
                        str(train_id)
                    ])

                    if trial_info['has_dmd']:
                        dmd_burst_tick += 1
                    if trial_info['has_laser']:
                        laser_burst_tick

                burst_offset += burst_count
                if trial_info['has_dmd']:
                    dmd_tick += 1
                if trial_info['has_laser']:
                    laser_tick += 1

            # dtype & structured array
            maxlen = max(len(b[6]) for b in bursts_list)
            dtype = np.dtype([
                ("dmd_train_onset", "f4"),
                ("dmd_burst_onset", "f4"),
                ("dmd_burst_offset", "f4"),
                ("laser_train_onset", "f4"),
                ("laser_burst_onset", "f4"),
                ("laser_burst_offset", "f4"),
                ("train_id", f"S{maxlen}")
            ])

            triggers_array = np.zeros(len(bursts_list), dtype=dtype)
            for i, b in enumerate(bursts_list):
                triggers_array[i] = (b[0], b[1], b[2], b[3], b[4], b[5], b[6].encode("utf-8"))

            rec_grp.create_dataset("triggers", data=triggers_array,
                                compression="gzip", chunks=True)

            # -----------------------------
            # 1b) Trial info
            # -----------------------------
            dtype_fields = []
            for col in train_rec_df.columns:
                if train_rec_df[col].dtype.kind in "i":
                    dtype_fields.append((col, "i4"))
                elif train_rec_df[col].dtype.kind in "f":
                    dtype_fields.append((col, "f4"))
                else:
                    maxlen_col = train_rec_df[col].astype(str).map(len).max()
                    dtype_fields.append((col, f"S{maxlen_col}"))

            table_array = np.zeros(len(train_rec_df), dtype=np.dtype(dtype_fields))
            for i, (_, row) in enumerate(train_rec_df.iterrows()):
                for col in train_rec_df.columns:
                    val = row[col]
                    if isinstance(val, str):
                        table_array[i][col] = val.encode("utf-8")
                    elif pd.isna(val):
                        table_array[i][col] = np.nan if train_rec_df[col].dtype.kind in "f" else -1
                    else:
                        table_array[i][col] = val

            rec_grp.create_dataset("trial_info", data=table_array,
                                compression="gzip", chunks=True)

            # -----------------------------
            # 1c) Per-recording cluster data (spiketimes + waveforms)
            # -----------------------------
            clusters_grp = rec_grp.require_group("clusters")
            for cluster_id, cinfo in cluster_info.iterrows():
                cluster_rec_grp = clusters_grp.require_group(str(cluster_id))

                # Access the correct key for spiketimes (patch for 250904_A)
                rec_nr = rec_id.split('_')[2]
                spiketimes_key = None
                for k in spiketimes.keys():
                    if rec_nr in k:
                        spiketimes_key = k
                        break

                assert spiketimes_key is not None
 
                cluster_rec_grp.create_dataset('spiketimes', data=spiketimes[spiketimes_key][cluster_id])
                if include_waveforms:
                    cluster_rec_grp.create_dataset('waveforms', data=waveforms[spiketimes_key][cluster_id])

    print(f'\nSaved dataset to {write_file.as_posix()}')
