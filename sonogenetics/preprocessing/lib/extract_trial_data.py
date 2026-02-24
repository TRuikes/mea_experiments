import pandas as pd
from sonogenetics.preprocessing.lib.filepaths import FilePaths
from sonogenetics.preprocessing.dataset_sessions import dataset_sessions
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
            df_read['rec_file'] = f
            dataframes.append(df_read)
        df = pd.concat(dataframes, ignore_index=True)

    if df.shape[1] == 0:  # use another delimiter
        df = pd.read_csv(
            filepaths.raw_trials, index_col=0, header=0,
            delimiter=';'
        )

    train_i = 0
    df = df.reset_index(drop = True)

    for i, r in df.iterrows():

        tid = f'tid_{filepaths.sid}_{train_i:03d}'
        df.at[i, 'train_id_index'] = tid
        df.at[i, 'train_id'] = tid
        train_i += 1

        rec_nr = r['Recording Number']
        recording_name = None

        for rr in filepaths.recording_names:
            if f'_{rec_nr:01.0f}_' in rr:
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
    mea_position = pd.read_csv(filepaths.mea_position_file, index_col=0, header=0)
    for i, r in df.iterrows():
        if pd.isna(r.electrode):
            if 'dmd' or 'light' in r.protocol:
                continue
            else:
                raise ValueError('error??')
        p = mea_position.loc[r.electrode]
        df.at[i, 'laser_x'] = p.x
        df.at[i, 'laser_y'] = p.y

    # try:
    try:
        laser_specs = pd.read_csv(filepaths.laser_calib_file, index_col=0, header=0)
        laser_found = True
    except:
        laser_found = False
        laser_specs = None

    for i, r in df.iterrows():
        if 'protocol' in r.keys():
            sequence_name = r.protocol
        else:
            sequence_name = r.sequence_name

        if sequence_name in [
            'pa_prr_series', 'pa_light_prr_series', 'pilot_stimparams', 'pa_dose_sequence_1',
        ]:
            
            cb = r['Connected Fibers']
            
            att = r['Attenuators']

            if att == 5.8 or att == '5.8':
                fiber_connection = f'oem_58_{cb}'
            else:
                raise ValueError('implement this')

            if 'laser_power' in r.keys():
                dac_voltage = r.laser_power
            else:
                dac_voltage = r.dac_voltage

            df.at[i, 'dac_voltage'] = dac_voltage
            df.at[i, 'pulse_repetition_rate'] = r.laser_pulse_repetition_rate

            if '_C6' in fiber_connection or '_C7' in fiber_connection:
                diameter = 50 / 1e3  # mm
                large_diameter = 100 / 1e3  # mm
            elif '_C2' in fiber_connection:
                diameter = 200 / 1e3  # mm
                large_diameter = 300 / 1e3  # mm
            elif '_C8' in fiber_connection:
                diameter = 25  / 1e3  # mm
                large_diameter = 50 / 1e3  # mm
            else:
                raise ValueError(f'implement this: {fiber_connection}')


            area = np.pi * (diameter / 2) ** 2
            large_area = np.pi * (large_diameter / 2) ** 2

            df.at[i, 'fiber_diameter'] = diameter
            # df.at[i, 'irradiance'] = irradiance
            # df.at[i, 'irradiance_exact_fiber_diameter'] = irradiance  # irradiane at exact fiber diameter
            # df.at[i, 'irradiance_large_fiber_diameter'] = power / large_area  # W / mm2

    for i, r in df.iterrows():
        if 'dmd_burst_duration' in r.keys() and pd.notna(r.dmd_burst_duration) and r.dmd_burst_duration > 0:
            df.at[i, 'has_dmd'] = True
        else:
            df.at[i, 'has_dmd'] = False
        
        if 'laser_burst_duration' in r.keys() and pd.notna(r.laser_burst_duration) and r.laser_burst_duration > 0:
            df.at[i, 'has_laser'] = True
        else:
            df.at[i, 'has_laser'] = False
            
    df.to_csv(filepaths.proc_pp_trials)