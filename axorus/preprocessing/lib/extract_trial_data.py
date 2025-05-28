import pandas as pd
from axorus.preprocessing.lib.filepaths import FilePaths
from axorus.preprocessing.dataset_sessions import dataset_sessions
import numpy as np


def extract_trial_data(filepaths: FilePaths):

    if len(filepaths.raw_trials) == 1:
        df = pd.read_csv(filepaths.raw_trials[0], index_col=0, header=0)
    else:
        dataframes = []
        for f in filepaths.raw_trials:
            df_read = pd.read_csv(f, index_col=0, header=0)
            if df_read.shape[1] == 0:
                df_read = pd.read_csv(f, index_col=0, header=0, delimiter=';')
            df_read = df_read.loc[pd.notna(df_read.index)]
            dataframes.append(df_read)
        df = pd.concat(dataframes, ignore_index=True)

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
        train_i += 1

        if pd.isna(r.protocol):
            print(i)

        if 'dmd' in r.protocol:
            tail = '_dmd'
        elif 'pa' in r.protocol:
            tail = '_pa'
        else:
            raise ValueError('dunno')

        rec_nr = r.recording_number
        blocker = r.blockers
        recording_name = None

        for rr in filepaths.recording_names:
            if f'_{rec_nr:03.0f}_' in rr:
                recording_name = rr
        assert recording_name is not None

        df.at[i, 'recording_name'] = recording_name

    df.set_index('train_id_index', inplace=True)

    for rec_name, df_rec in df.groupby('recording_name'):
        ri = 0
        for i, r in df_rec.iterrows():
            df.at[i, 'rec_train_i'] = ri
            ri += 1

    # Add x,y position of laser to data
    mea_position = pd.read_csv(filepaths.raw_mea_position, index_col=0, header=0)
    for i, r in df.iterrows():
        if pd.isna(r.electrode):
            if 'dmd' in r.protocol:
                continue
            else:
                raise ValueError('error??')
        p = mea_position.loc[r.electrode]
        df.at[i, 'laser_x'] = p.x
        df.at[i, 'laser_y'] = p.y

    # try:
    laser_specs = pd.read_csv(filepaths.laser_calib_file, index_col=0, header=0)

    for i, r in df.iterrows():
        if 'dmd' in r.protocol:
            continue

        if filepaths.sid == '161024_A':
            fiber_connection = dataset_sessions[filepaths.sid]['fiber_connection']
        elif 'connected_fiber' in df.columns:
            cb = r.connected_fiber
            att = r.attenuator

            if att == 1.4 or att == '1.4':
                fiber_connection = f'CB1_14_{cb}'
            elif att == 'CA':
                fiber_connection = f'CB1_CA_{cb}_{r.n_turns:.0f}'
            else:
                raise ValueError('implement this')
        else:
            raise ValueError('implement this')

        laser_level = r.laser_level
        duty_cycle = r.duty_cycle

        # I did not fit the slope/intercept in 2D if I recorded the power for
        # too few sessions

        if fiber_connection == 'CB1_CA_C7_23':
            continue


        if 'slope_slope' in laser_specs.loc[fiber_connection].keys():
            slope_slope = laser_specs.loc[fiber_connection]['slope_slope']
            slope_intercept = laser_specs.loc[fiber_connection]['slope_intercept']
            inter_slope = laser_specs.loc[fiber_connection]['inter_slope']
            inter_intercept = laser_specs.loc[fiber_connection]['inter_intercept']
            fr_slope_slope = laser_specs.loc[fiber_connection]['fr_slope_slope']
            fr_slope_intercept = laser_specs.loc[fiber_connection]['fr_slope_intercept']
            fr_inter_slope = laser_specs.loc[fiber_connection]['fr_inter_slope']
            fr_inter_intercept = laser_specs.loc[fiber_connection]['fr_inter_intercept']

            power_slope = slope_intercept + slope_slope * laser_level
            power_inter = inter_intercept + inter_slope * laser_level

        else:

            power_slope = laser_specs.loc[fiber_connection]['power_slope']
            power_inter = laser_specs.loc[fiber_connection]['power_intercept']
            fr_slope_slope = laser_specs.loc[fiber_connection]['fr_slope_slope']
            fr_slope_intercept = laser_specs.loc[fiber_connection]['fr_slope_intercept']
            fr_inter_slope = laser_specs.loc[fiber_connection]['fr_inter_slope']
            fr_inter_intercept = laser_specs.loc[fiber_connection]['fr_inter_intercept']

        power = power_inter + power_slope * duty_cycle  # mW

        frep_slope = fr_slope_intercept + fr_slope_slope * laser_level  # Hz
        frep_inter = fr_inter_intercept + fr_inter_slope * laser_level

        frep = frep_inter + frep_slope * duty_cycle

        df.at[i, 'laser_power'] = power
        df.at[i, 'repetition_frequency'] = frep
        df.at[i, 'e_pulse'] = ((power / 1000) / frep) * 1e6

        if '_C6' in fiber_connection or '_C7':
            diameter = 50 / 1e3  # mm
            large_diameter = 100 / 1e3  # mm
        elif '_C8' in fiber_connection:
            diameter = 25  / 1e3  # mm
            large_diameter = 50 / 1e3  # mm
        else:
            raise ValueError(f'implement this: {fiber_connection}')


        area = np.pi * (diameter / 2) ** 2
        large_area = np.pi * (large_diameter / 2) ** 2
        power = power / 1000  # W
        irradiance = power / area  # W / mm2

        df.at[i, 'fiber_diameter'] = diameter
        df.at[i, 'irradiance'] = irradiance
        df.at[i, 'irradiance_exact_fiber_diameter'] = irradiance  # irradiane at exact fiber diameter
        df.at[i, 'irradiance_large_fiber_diameter'] = power / large_area  # W / mm2

    # except:
    #     print(f'ERROR IN LASER POWER DATAFRAME, FIX THIS')

    df.to_csv(filepaths.proc_pp_trials)