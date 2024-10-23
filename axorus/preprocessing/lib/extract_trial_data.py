import pandas as pd
from axorus.preprocessing.lib.filepaths import FilePaths
from axorus.preprocessing.dataset_sessions import dataset_sessions
import numpy as np


def extract_trial_data(filepaths: FilePaths):

    df = pd.read_csv(filepaths.raw_trials, index_col=0, header=0)

    if df.shape[1] == 0:  # use another delimiter
        df = pd.read_csv(
            filepaths.raw_trials, index_col=0, header=0,
            delimiter=';'
        )

    train_i = 0

    for i, r in df.iterrows():

        tid = f'tid_{filepaths.sid}_{train_i:03d}'
        df.at[i, 'train_id_index'] = tid
        df.at[i, 'train_id'] = tid
        df.at[i, 'train_i'] = int(train_i)
        train_i += 1

    df.set_index('train_id_index', inplace=True)

    # Add x,y position of laser to data
    mea_position = pd.read_csv(filepaths.raw_mea_position, index_col=0, header=0)
    for i, r in df.iterrows():
        p = mea_position.loc[r.electrode]
        df.at[i, 'laser_x'] = p.x
        df.at[i, 'laser_y'] = p.y

    # Add laser stim specs to data
    laser_specs = pd.read_csv(filepaths.laser_calib_file, index_col=0, header=0)

    for i, r in df.iterrows():
        if filepaths.sid == '161024_A':
            fiber_connection = dataset_sessions[filepaths.sid]['fiber_connection']
        else:
            raise ValueError('implement this')

        slope_slope = laser_specs.loc[fiber_connection]['slope_slope']
        slope_intercept = laser_specs.loc[fiber_connection]['slope_intercept']
        inter_slope = laser_specs.loc[fiber_connection]['inter_slope']
        inter_intercept = laser_specs.loc[fiber_connection]['inter_intercept']
        fr_slope_slope = laser_specs.loc[fiber_connection]['fr_slope_slope']
        fr_slope_intercept = laser_specs.loc[fiber_connection]['fr_slope_intercept']
        fr_inter_slope = laser_specs.loc[fiber_connection]['fr_inter_slope']
        fr_inter_intercept = laser_specs.loc[fiber_connection]['fr_inter_intercept']

        laser_level = r.laser_level
        duty_cycle = r.duty_cycle

        power_slope = slope_intercept + slope_slope * laser_level
        power_inter = inter_intercept + inter_slope * laser_level

        power = power_inter + power_slope * duty_cycle

        frep_slope = fr_slope_intercept + fr_slope_slope * laser_level
        frep_inter = fr_inter_intercept + fr_inter_slope * laser_level

        frep = frep_inter + frep_slope * duty_cycle

        df.at[i, 'laser_power'] = power
        df.at[i, 'repetition_frequency'] = frep
        df.at[i, 'e_pulse'] = ((power / 1000) / frep) * 1e6

        if '_C6' in fiber_connection:
            diameter = 50 / 1e3  # mm
        else:
            diameter = 200 / 1e3  # mm

        area = np.pi * (diameter / 2) ** 2
        power = power / 1000  # W
        irradiance = power / area  # W / mm2

        df.at[i, 'irradiance'] = irradiance

    df.to_csv(filepaths.proc_pp_trials)