from sonogenetics.preprocessing.params import (data_sample_rate, data_type, data_nb_channels,
                                         data_trigger_channels, data_voltage_resolution,
                                         data_trigger_thresholds)
from sonogenetics.preprocessing.lib.filepaths import FilePaths
import utils
from tqdm import tqdm
import numpy as np
import pandas as pd


def align_trials_and_triggers(filepaths: FilePaths, update=False,
                              recording_numbers_to_skip=None):
    print('\nAligning trigger data')

    if filepaths.proc_pp_triggers.exists() and not update:
        print(f'\ttriggers already extracted')
        return

    data = utils.load_nested_dict(filepaths.proc_pp_triggers)

    trials = pd.read_csv(filepaths.proc_pp_trials, index_col=0, header=0)

    for rec_id, trials_rec in trials.groupby('recording_name'):

        SKIP_RECORDING = False
        if recording_numbers_to_skip is not None:
            for nr in recording_numbers_to_skip:
                if f'_{nr:.0f}_' in rec_id:
                    print(f'\tskipping recording {rec_id}')
                    SKIP_RECORDING = True

        if SKIP_RECORDING:
            continue


        # The dmd trigger is not corrupted
        dmd_train_onsets = data[rec_id]['dmd']['train_onsets']
        laser_train_onsets = data[rec_id]['laser']['train_onsets']
        assert dmd_train_onsets.shape[0] == trials_rec.has_dmd.sum()

        dmd_tick = 0

        n_trials = len(trials_rec)
        for trial_i in range(n_trials):

            df_idx = trials_rec.index.values[trial_i]
            has_laser = trials_rec.iloc[trial_i]['has_laser']
            has_dmd = trials_rec.iloc[trial_i]['has_dmd']

            if has_dmd:
                dmd_train_onset = dmd_train_onsets[dmd_tick]
                trials.at[df_idx, 'a_dmd_train_onset'] = dmd_train_onset

                dmd_tick += 1

            if has_dmd and has_laser:
                idx = np.where(np.abs(laser_train_onsets - dmd_train_onset) < 1e4)[0]

                if len(idx) == 1:
                    trials.at[df_idx, 'a_laser_train_onset'] = laser_train_onsets[idx[0]]
                    trials.at[df_idx, 'aligned_laser'] = True
                else:
                    trials.at[df_idx, 'aligned_laser'] = False

            elif has_dmd and not has_laser:
                trials.at[df_idx, 'aligned_laser'] = True


    for rec_id, trials_rec in trials.groupby('recording_name'):

        SKIP_RECORDING = False
        if recording_numbers_to_skip is not None:
            for nr in recording_numbers_to_skip:
                if f'_{nr:.0f}_' in rec_id:
                    print(f'\tskipping recording {rec_id}')
                    SKIP_RECORDING = True

        if SKIP_RECORDING:
            continue

        # The dmd trigger is not corrupted
        laser_train_onsets = data[rec_id]['laser']['train_onsets']
        n_trials = len(trials_rec)


        for trial_i in range(n_trials):

            df_idx = trials_rec.index.values[trial_i]
            has_laser = trials_rec.iloc[trial_i]['has_laser']
            laser_train_onset = trials_rec.iloc[trial_i]['a_laser_train_onset']

            if has_laser and pd.isna(laser_train_onset):
                to_prev = trials_rec.iloc[trial_i-1]['a_laser_train_onset']
                to_next = trials_rec.iloc[trial_i+1]['a_laser_train_onset']
                if pd.isna(to_prev) or pd.isna(to_next):
                    continue

                idx = np.where((laser_train_onsets > to_prev) & (laser_train_onsets < to_next))[0]
                if len(idx) == 0:
                    continue
                elif len(idx) == 1:
                    trials.at[df_idx, 'a_laser_train_onset'] = laser_train_onsets[idx[0]]
                    trials.at[df_idx, 'aligned_laser'] = True
                else:
                    continue

    trials.to_csv(filepaths.prop_pp_aligned_trials)


