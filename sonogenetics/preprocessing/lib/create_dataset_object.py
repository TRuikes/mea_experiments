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
    Converts a pandas DataFrame with mixed float/text/NaN columns into a
    NumPy structured array compatible with HDF5 datasets.

    - Float NaNs are filled with -99.0
    - Text NaNs are filled with empty strings
    """
    # 1. Identify text and float columns
    text_cols = [col for col in df.columns if df[col].dtype == "object" or isinstance(df[col].iloc[0], str)]
    float_cols = [col for col in df.columns if col not in text_cols]

    # 2. Handle NaNs cleanly without modifying the original DataFrame
    df_filled = df.copy()
    df_filled[float_cols] = df_filled[float_cols].fillna(-99.0)
    df_filled[text_cols] = df_filled[text_cols].fillna("")

    # 3. Dynamically determine maximum string byte-lengths for text columns
    max_lens = {}
    for col in text_cols:
        # Calculate max byte length after encoding to utf-8 (fallback to 1 if empty)
        max_bytes = df_filled[col].astype(str).str.encode("utf-8").str.len().max()
        max_lens[col] = max(max_bytes, 1)

    # 4. Dynamically build the structured dtype list
    dtype_list = []
    for col in df.columns:
        if col in text_cols:
            dtype_list.append((str(col), f"S{max_lens[col]}"))
        else:
            dtype_list.append((str(col), "f4"))

    dtype = np.dtype(dtype_list)

    # 5. Create and populate the structured array using fast column-vectorization
    structured_array = np.zeros(len(df_filled), dtype=dtype)

    for col in df.columns:
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
        # # Determine max length of index if it's string/object
        # if cluster_info.index.dtype.kind in "O" or cluster_info.index.dtype.kind in "U" or cluster_info.index.dtype.kind in "S":
        #     maxlen_index = cluster_info.index.astype(str).map(len).max()
        #     index_dtype = f"S{maxlen_index}"
        # else:  # numeric index
        #     index_dtype = "i8"
        #
        # # Add index as first field
        # cluster_dtype_fields = [("index", index_dtype)]
        # for col in cluster_info.columns:
        #     if col == 'group':
        #         continue
        #
        #     if cluster_info[col].dtype.kind in "i":
        #         cluster_dtype_fields.append((col, "i4"))
        #     elif cluster_info[col].dtype.kind in "f":
        #         cluster_dtype_fields.append((col, "f4"))
        #     else:
        #         maxlen_col = cluster_info[col].astype(str).map(len).max()
        #         cluster_dtype_fields.append((col, f"S{maxlen_col}"))
        #
        # # Create structured array
        # cluster_table = np.zeros(len(cluster_info), dtype=np.dtype(cluster_dtype_fields))
        #
        # for i, (idx, row) in enumerate(cluster_info.iterrows()):
        #     # Store original index
        #     if isinstance(idx, str):
        #         cluster_table[i]["index"] = idx.encode("utf-8")
        #     else:
        #         cluster_table[i]["index"] = idx
        #
        #     for col in cluster_info.columns:
        #         if col == 'group':
        #             continue
        #
        #         val = row[col]
        #         if isinstance(val, str):
        #             cluster_table[i][col] = val.encode("utf-8")
        #         elif pd.isna(val):
        #             cluster_table[i][col] = np.nan if cluster_info[col].dtype.kind in "f" else -1
        #         else:
        #             cluster_table[i][col] = val

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

                # Patching corrupted data
                if rec_nr == 6 and filepaths.sid == '2026-06-30 rat LE 803 Mekano6 A':
                    if laser_tick == 43:
                        break
                if rec_id == 'rec_5_B_20260630_dmd_full_field_intensities':
                    if dmd_tick > 80:
                        break

                # Checking if there are laser and dmd
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
                        dmd_burst_onsets[dmd_burst_tick]  if trial_info['has_dmd'] else -1,
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

            # valid_columns = [
            #     col for col in train_rec_df.columns
            #     if not pd.isna(train_rec_df[col]).all()
            # ]
            #
            # dtype_fields = []
            # for col in valid_columns:
            #     col_data = train_rec_df[col]
            #
            #     if col_data.dtype.kind in "i":
            #         dtype_fields.append((col, "i4"))
            #     elif col_data.dtype.kind in "f":
            #         dtype_fields.append((col, "f4"))
            #     elif col_data.dtype.kind == 0:
            #         # Handle the new data.kind type = '0'
            #         # Option A: If you want it treated as a string/object:
            #         maxlen_col = col_data.astype(str).map(len).max()
            #         dtype_fields.append((col, f"S{maxlen_col}"))
            #     else:
            #         # handle object/string columns
            #         print(col_data.dtype.kind)
            #         maxlen_col = col_data.astype(str).map(len).max()
            #         dtype_fields.append((col, f"S{maxlen_col}"))
            #
            # table_array = np.zeros(len(train_rec_df), dtype=np.dtype(dtype_fields))
            #
            # # -------------------------------
            # # Fill array
            # # -------------------------------
            # for i, (_, row) in enumerate(train_rec_df.iterrows()):
            #     for col in valid_columns:
            #         val = row[col]
            #         col_dtype = train_rec_df[col].dtype.kind
            #
            #         if isinstance(val, str):
            #             table_array[i][col] = val.encode("utf-8")
            #
            #         elif pd.isna(val):
            #             if col_dtype in "f":
            #                 table_array[i][col] = np.nan
            #             elif col_dtype in "i":
            #                 table_array[i][col] = -1
            #             else:
            #                 table_array[i][col] = b""  # empty string for object
            #
            #         else:
            #             table_array[i][col] = val

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
