import pandas as pd
from sonogenetics.analysis.lib.data_io import DataIO
from typing import Dict

params_per_protocol = {
    # 2026-02-11 recording
    'rec_2_pilot_021126': ['dac_voltage', 'laser_pulse_repetition_rate', 'laser_burst_duration'],
    'rec_3_pilot_021126_cppcnqx': ['dac_voltage', 'laser_pulse_repetition_rate', 'laser_burst_duration'],

    'pa_dmd_pilot1': ['laser_onset_delay'],
    'pa_dose_sequence_1': ['dac_voltage', 'laser_pulse_repetition_rate', 'laser_burst_duration'],
    'rec_1_B_20260325_pa_intensity_test': ['laser_power', 'laser_pulse_repetition_rate', 'laser_burst_duration'],
    'rec_2_A_20260325_pa_intensity_test': ['laser_power', 'laser_pulse_repetition_rate', ],
    'rec_2_B_20260325_pa_dmd_timing': ['laser_onset_delay', 'dmd_onset_delay'],
    'rec_3_A_20260325_pa_dmd_timing': ['laser_onset_delay', 'dmd_onset_delay'],
    'rec_3_B_20260325_dmd_full_field': ['dmd_onset_delay'],
    'rec_4_A_20260325_dmd_full_field': ['dmd_onset_delay'],
    'pilot_stimparams': ['dac_voltage', 'laser_pulse_repetition_rate', 'laser_burst_duration'],

}

params_abbreviation = {
    'laser_onset_delay':'l-del',
    'laser_burst_duration': 'l-bd',
    'laser_pulse_repetition_rate': 'l-pp',
    'dac_voltage': 'l-pwr',
    'laser_power': 'l-pwr',
    'dmd_onset_delay': 'd-del',
}


def detect_preferred_electrode(data_io: DataIO, cells_df: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    # %% Detect electrode stim site with most significant responses, per cell
    output = {}

    # Detect per recording + protocol pair the most responsive stimulation site for each cell
    # Most responsive stimsite = electrode with most significant responses.
    for (rec_id, protocol), rdf in data_io.train_df.groupby(['rec_id', 'protocol']):
        pref_ec_df = pd.DataFrame()

        for cluster_id in cells_df.index.tolist():

            pref_ec = None
            n_sig_pref_ec = None

            for ec, edf in rdf.groupby('electrode'):
                n_sig = 0

                for tid in edf.index.values:
                    if cells_df.loc[cluster_id, (tid, 'is_excited')] == True or cells_df.loc[cluster_id, (tid, 'is_inhibited')] == True:
                        n_sig += 1

                if n_sig > 0:
                    if pref_ec is None or n_sig > n_sig_pref_ec:
                        pref_ec = ec
                        n_sig_pref_ec = n_sig

            pref_ec_df.at[cluster_id, 'ec'] = pref_ec

        if rec_id not in output:
            output[rec_id] = {}

        output[rec_id][protocol] = pref_ec_df

    return output


if __name__ == '__main__':
    from sonogenetics.analysis.lib.data_io import DataIO
    from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis

    session_id = '2026-02-11 mouse c57 565 eMSCL A'
    dio = DataIO(dataset_dir)
    dio.load_session(session_id, load_pickle=True, load_waveforms=False)

    loadname = dataset_dir / f'{session_id}_cells.csv'
    cdf = pd.read_csv(loadname, header=[0, 1], index_col=0)

    ec_preference = detect_preferred_electrode(dio, cdf)
