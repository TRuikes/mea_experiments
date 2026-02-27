import pandas as pd
from sonogenetics.analysis.data_io import DataIO


params_per_protocol = {
    'pa_dmd_pilot1': ['laser_onset_delay'],
    'pa_dose_sequence_1': ['dac_voltage', 'laser_pulse_repetition_rate', 'laser_burst_duration'],
}

params_abbreviation = {
    'laser_onset_delay':'l-del',
    'laser_burst_duration': 'l-bd',
    'laser_pulse_repetition_rate': 'l-pp',
    'dac_voltage': 'l-pwr',
}


def detect_preferred_electrode(data_io: DataIO, cells_df: pd.DataFrame):
    # %% Detect electrode stim site with most significant responses, per cell
    output = {}

    protocols = data_io.train_df.protocol.unique()

    for protocol in protocols:
        pref_ec_df = pd.DataFrame()

        for cluster_id in cells_df.index.tolist():


            pref_ec = None
            n_sig_pref_ec = None

            trials = data_io.train_df.query('protocol == @protocol')
            electrodes = trials.electrode.unique()

            for ec in electrodes:

                df = data_io.train_df.query(f'electrode == {float(ec)} and protocol == @protocol')
                n_sig = 0

                for tid in df.index.values:
                    if cells_df.loc[cluster_id, (tid, 'is_excited')] == True or cells_df.loc[cluster_id, (tid, 'is_inhibited')] == True:
                        n_sig += 1

                if n_sig > 0:
                    if pref_ec is None or n_sig > n_sig_pref_ec:
                        pref_ec = ec
                        n_sig_pref_ec = n_sig
                    # elif n_sig == n_sig_pref_ec:
                    #     print(f'cluster {cluster_id} has 2 pref ecs')

            pref_ec_df.at[cluster_id, 'ec'] = pref_ec


        output[protocol] = pref_ec_df


    return output


