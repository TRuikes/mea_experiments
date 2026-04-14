import pandas as pd
import utils

manual_curated_ids = [
    'rec_3_B_20260325_dmd_full_field',
    # 'rec_2_A_20260317_pa_intensity_test',
]


def check_recording_and_dataframe_match(filepaths, recording_numbers_to_skip):

    train_df = pd.read_csv(filepaths.proc_pp_trials, index_col=0, header=0)
    triggers = utils.load_nested_dict(filepaths.proc_pp_triggers)


    for rec_id in filepaths.recording_names:
        if rec_id in manual_curated_ids:
            continue

        # Exclude this recording if listed so in dataset_sessions
        rec_nr = int(rec_id.split('_')[1])
        if rec_nr in recording_numbers_to_skip:
            continue

        # Verify that the triggers detected in the data match those with the dataframes
        has_triggers = False

        train_df_check = train_df.loc[train_df['Recording Number'] == rec_nr]

        if 'PA' in rec_id or 'buSTIM1' in rec_id or 'pilot_021126' in rec_id or 'pilot021626' in rec_id or 'pa' in rec_id:
            pa_train_onsets = triggers[rec_id]['laser']['train_onsets']
            print(f'{rec_id} - len train onsets: {len(pa_train_onsets)}')
            pa_n_train_recording = len(pa_train_onsets)

            pa_burst_onsets = triggers[rec_id]['laser']['burst_onsets']
            pa_n_bursts_triggers = len(pa_burst_onsets)

            print(f'\tAdding PA triggers')
            has_triggers = True

            pa_n_train_dataframe = train_df_check.has_laser.sum()
            assert pa_n_train_recording == pa_n_train_dataframe, f'{filepaths.sid}'

            pa_n_bursts_train_df = train_df_check.laser_burst_count.sum()
            assert pa_n_bursts_triggers == pa_n_bursts_train_df, f'{filepaths.sid}, {pa_n_bursts_triggers}, {pa_n_train_dataframe}'

        if 'DMD' in rec_id or 'chirp' in rec_id or 'dmd' in rec_id:
            dmd_train_onsets = triggers[rec_id]['dmd']['train_onsets']
            dmd_n_trials_triggers = len(dmd_train_onsets)

            dmd_burst_onsets = triggers[rec_id]['dmd']['burst_onsets']
            dmd_n_bursts_triggers = len(dmd_burst_onsets)

            print(f'\tAdding DMD triggers')
            has_triggers = True

            # Check DMD triggers

            assert dmd_n_trials_triggers == train_df_check.shape[0]
            assert dmd_n_bursts_triggers == train_df_check.dmd_burst_count.sum()

        assert has_triggers