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


def df_to_hdf5_structured_array(df: pd.DataFrame) -> np.ndarray:
    """
    Converts a pandas DataFrame with mixed float/text/NaN columns,
    including its index, into a NumPy structured array compatible with HDF5 datasets.

    - Float NaNs are filled with -99.0
    - Text NaNs are filled with empty strings
    """
    # 1. Bring the index into the DataFrame as a regular column to process it uniformly
    df_filled = df.copy()
    index_name = df.index.name if df.index.name is not None else "index"

    # Insert index at the front of our working dataframe copy
    df_filled.insert(0, index_name, df.index)

    # 2. Identify text and float columns (now including the index column)
    text_cols = [col for col in df_filled.columns if
                 df_filled[col].dtype == "object" or (len(df_filled) > 0 and isinstance(df_filled[col].iloc[0], str))]
    float_cols = [col for col in df_filled.columns if col not in text_cols]

    # 3. Handle NaNs cleanly without modifying the original DataFrame
    df_filled[float_cols] = df_filled[float_cols].fillna(-99.0)
    df_filled[text_cols] = df_filled[text_cols].fillna("")

    # 4. Dynamically determine maximum string byte-lengths for text columns
    max_lens = {}
    for col in text_cols:
        # Calculate max byte length after encoding to utf-8 (fallback to 1 if empty)
        max_bytes = df_filled[col].astype(str).str.encode("utf-8").str.len().max()
        max_lens[col] = max(max_bytes, 1) if pd.notna(max_bytes) else 1

    # 5. Dynamically build the structured dtype list
    dtype_list = []
    for col in df_filled.columns:
        if col in text_cols:
            dtype_list.append((str(col), f"S{max_lens[col]}"))
        # If the index or column is strictly an integer, we can preserve it as i4/i8,
        # otherwise we fallback to your default f4 type.
        elif np.issubdtype(df_filled[col].dtype, np.integer):
            dtype_list.append((str(col), "i8"))
        else:
            dtype_list.append((str(col), "f4"))

    dtype = np.dtype(dtype_list)

    # 6. Create and populate the structured array using fast column-vectorization
    structured_array = np.zeros(len(df_filled), dtype=dtype)

    for col in df_filled.columns:
        if col in text_cols:
            structured_array[str(col)] = df_filled[col].astype(str).str.encode("utf-8")
        else:
            structured_array[str(col)] = df_filled[col].values

    return structured_array


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

    write_file = filepaths.dataset_file_waveforms if include_waveforms else filepaths.dataset_file

    if not write_file.parent.exists():
        write_file.parent.mkdir(parents=True)

    with h5py.File(write_file, "w") as f:

        # -----------------------------
        # 0) Top-level cluster info table
        # -----------------------------
        cluster_table = df_to_hdf5_structured_array(cluster_info)

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

            # Exclude this recording if listed so in dataset_sessions
            rec_nr = int(rec_id.split('_')[1])
            if rec_nr in recording_numbers_to_skip:
                continue

            print(f"Loading {rec_id}")
            train_rec_df = train_df.loc[train_df['Recording Number'] == rec_nr]
            if train_rec_df.empty:
                continue

            if rec_id == 'rec_3_B_20260325_dmd_full_field':
                train_rec_df = train_rec_df.iloc[1:]

            # Group for this recording
            rec_grp = f.require_group(f"recordings/{rec_id}")

            # -----------------------------
            # 1a) Triggers array
            # -----------------------------
            bursts_list = []
            burst_offset = 0

            if train_rec_df['has_laser'].sum() > 0:
                laser_train_onsets = triggers[rec_id]["laser"]["train_onsets"]
                laser_burst_onsets = triggers[rec_id]["laser"]["burst_onsets"]
                laser_burst_offsets = triggers[rec_id]["laser"]["burst_offsets"]
            else:
                laser_train_onsets = None
                laser_burst_onsets = None
                laser_burst_offsets = None

            if train_rec_df['has_dmd'].sum() > 0:
                dmd_train_onsets = triggers[rec_id]["dmd"]["train_onsets"]
                dmd_burst_onsets = triggers[rec_id]["dmd"]["burst_onsets"]
                dmd_burst_offsets = triggers[rec_id]["dmd"]["burst_offsets"]

                if rec_id == 'rec_3_B_20260325_dmd_full_field':
                    dmd_train_onsets = dmd_train_onsets[1:]
                    dmd_burst_onsets = dmd_burst_onsets[8:]
                    dmd_burst_offsets = dmd_burst_offsets[8:]

            else:
                dmd_train_onsets = None
                dmd_burst_onsets = None
                dmd_burst_offsets = None

            # Ticker to index into laser and dmd trigger onsets
            # This becomes relevant if in 1 recording there are trials with
            # without dual stimulation. In which case there are more
            # dmd or laser triggers
            # The ticks are a bit redundant, but the could would break if there
            # are fewer detected triggers than trials, so its a nice backup
            dmd_tick, laser_tick = 0, 0
            dmd_burst_tick, laser_burst_tick = 0, 0

            for train_id, trial_info in train_rec_df.iterrows():
                laser_burst_count = trial_info['laser_burst_count'] if trial_info['has_laser'] else 0
                dmd_burst_count = trial_info['dmd_burst_count'] if trial_info['has_dmd'] else 0

                # Detect the number of bursts for this trial
                if trial_info['has_laser'] and trial_info['has_dmd']:
                    assert laser_burst_count == dmd_burst_count
                    burst_count = laser_burst_count
                elif trial_info['has_laser'] and not trial_info['has_dmd']:
                    burst_count = laser_burst_count
                elif trial_info['has_dmd'] and not trial_info['has_laser']:
                    burst_count = dmd_burst_count
                else:
                    raise ValueError('i should not have ended up here?')

                for burst_i in range(int(burst_count)):

                    bursts_list.append([
                        dmd_train_onsets[dmd_tick] if trial_info['has_dmd'] else -1,
                        dmd_burst_onsets[dmd_burst_tick] if trial_info['has_dmd'] else -1,
                        dmd_burst_offsets[dmd_burst_tick] if trial_info['has_dmd'] else -1,
                        laser_train_onsets[laser_tick] if trial_info['has_laser'] else -1,
                        laser_burst_onsets[laser_burst_tick] if trial_info['has_laser'] else -1,
                        laser_burst_offsets[laser_burst_tick] if trial_info['has_laser'] else -1,
                        str(train_id)
                    ])

                    if trial_info['has_dmd']:
                        dmd_burst_tick += 1
                    if trial_info['has_laser']:
                        laser_burst_tick += 1

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
            table_array = df_to_hdf5_structured_array(train_rec_df)

            rec_grp.create_dataset("trial_info", data=table_array,
                                compression="gzip", chunks=True)

            # -----------------------------
            # 1c) Per-recording cluster data (spiketimes + waveforms)
            # -----------------------------
            clusters_grp = rec_grp.require_group("clusters")
            for cluster_id, cinfo in cluster_info.iterrows():
                cluster_rec_grp = clusters_grp.require_group(str(cluster_id))
                cluster_rec_grp.create_dataset('spiketimes', data=spiketimes[rec_id][cluster_id])
                if include_waveforms:
                    cluster_rec_grp.create_dataset('waveforms', data=waveforms[rec_id][cluster_id])


    print(f'\nSaved dataset to {write_file.as_posix()}\n\n')
