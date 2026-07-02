from pv_chip_aarhus.preprocessing.lib.filepaths import FilePaths
import pandas as pd
import utils
import h5py
import numpy as np
import pickle

DATASAMPLE_RATE = 20000
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
                          ):
    print('\nCreating dataset object')
    train_df = pd.read_csv(filepaths.proc_pp_trials, index_col=0, header=0)

    # spiketimes = utils.load_nested_dict(filepaths.proc_pp_spiketimes)012
    #
    # if include_waveforms:
    #     waveforms = utils.load_nested_dict(filepaths.proc_pp_waveforms)
    #
    # triggers = utils.load_nested_dict(filepaths.proc_pp_triggers)
    # cluster_info = pd.read_csv(filepaths.proc_pp_clusterinfo, index_col=0, header=0)
    # mea_position = pd.read_csv(filepaths.mea_position_file, index_col=0, header=0)
    #
    # # Add cluster x and y to data
    # for i, r in cluster_info.iterrows():
    #     cluster_info.at[i, 'cluster_x'] = mea_position.loc[r.ch + 1].x
    #     cluster_info.at[i, 'cluster_y'] = mea_position.loc[r.ch + 1].y
    #
    write_file = filepaths.dataset_file_waveforms if include_waveforms else filepaths.dataset_file

    if not write_file.parent.exists():
        write_file.parent.mkdir(parents=True)

    with h5py.File(write_file, "w") as f_write:


        # Write trial dataframe to datasetobject
        write_array = df_to_hdf5_structured_array(train_df)
        f_write.create_dataset("trial_df", data=write_array,
                         compression="gzip", chunks=True)


        # -----------------------------
        # 1) Per recording data
        # -----------------------------

        for rec_nr, rec_info in filepaths.recording_table.iterrows():

            if 'flick' in rec_info.trigger_file:
                print('not adding flick recording')
                continue

            train_rec_df = train_df.loc[train_df['rec_nr'] == int(rec_nr)]
            if train_rec_df.empty:
                raise ValueError('')

            # Create a unique ID for this recording
            rec_id = f'{rec_nr}_{rec_info.lasermode}_{rec_info.stimsource}_{rec_info.varied_param}'

            # Create a datagroup in the HDF5 fil
            rec_grp = f_write.require_group(f"recordings/{rec_id}")

            # Load trigger data
            trigger_file = filepaths.trigger_dir / rec_info.trigger_file
            with open(trigger_file, "rb") as f:
                trigger_data = pickle.load(f)

            # Verify the number expected triggers match the number of measured triggers
            stimsource = train_rec_df.iloc[0]['stimsource']
            if stimsource == 'P' or stimsource == 'B':
                n_trials_in_stimfile = train_rec_df.pchr_repeats.sum()
                n_trials_in_trigger = len(trigger_data['poly']['pairs'])
                assert n_trials_in_trigger == n_trials_in_stimfile

            else:
                n_trials_in_stimfile = train_rec_df.laser_repeats.sum()
                n_trials_in_trigger = len(trigger_data['laser']['pairs'])
                assert n_trials_in_trigger == n_trials_in_stimfile


            # Now pair the trialmetadata with the triggers
            pchr_burst_trial_offset = 0  # keep track of where in the 'pairs' variable we are, relative to the
                # trial_info + ticker
            laser_burst_trial_offset = 0

            burst_df = pd.DataFrame()
            burst_i = 0

            for idx, trial_info in train_rec_df.iterrows():

                burst_offset = burst_i

                if stimsource in ['P', 'B']:

                    n_repeats = int(trial_info.pchr_repeats)

                    trial_nr = trial_info.train_id.split('_')[1]
                    for ti in range(n_repeats):
                        bid = f'bid_{trial_nr}_{trial_info.rec_nr}_{burst_i:03d}'
                        onset, offset = trigger_data['poly']['pairs'][pchr_burst_trial_offset + ti]

                        onset /= (DATASAMPLE_RATE / 1e3)  # in ms
                        offset /= (DATASAMPLE_RATE / 1e3)

                        burst_df.at[bid, 'pchr_onset'] = onset
                        burst_df.at[bid, 'pchr_offset'] = offset

                        # Make sure that the expected on duration matches the measured on duration
                        # with 0.5 ms accuracy
                        measured_on_duration = offset - onset
                        expected_on_duration = trial_info.pchr_on_duration
                        assert measured_on_duration - expected_on_duration < 0.5

                        for k, v in trial_info.items():
                            burst_df.at[bid, k] = v

                        burst_i += 1

                    pchr_burst_trial_offset += n_repeats

                if stimsource in ['L']:
                    burst_i = burst_offset # reset burst ticker to the value starting this trial, as we established
                        # that the number of bursts is always the same between polychrome and laser


                    n_repeats = int(trial_info.laser_repeats)

                    trial_nr = trial_info.train_id.split('_')[1]
                    for ti in range(n_repeats):
                        bid = f'bid_{trial_nr}_{trial_info.rec_nr}_{burst_i:03d}'

                        onset, offset = trigger_data['laser']['pairs'][laser_burst_trial_offset + ti]
                        onset /= (DATASAMPLE_RATE / 1e3)  # in ms
                        offset /= (DATASAMPLE_RATE / 1e3)

                        burst_df.at[bid, 'laser_onset'] = onset
                        burst_df.at[bid, 'laser_offset'] = offset

                        # Make sure that the expected on duration matches the measured on duration
                        # with 0.5 ms accuracy
                        measured_on_duration = offset - onset
                        expected_on_duration = trial_info.laser_on_duration

                        assert measured_on_duration - expected_on_duration < 1

                        for k, v in trial_info.items():
                            burst_df.at[bid, k] = v

                        burst_i += 1
                    laser_burst_trial_offset += n_repeats

            write_array = df_to_hdf5_structured_array(burst_df)

            # 6. Write to HDF5
            rec_grp.create_dataset("triggers_df", data=write_array,
                                   compression="gzip", chunks=True)

    print(f'\nSaved dataset to {write_file.as_posix()}\n\n')
