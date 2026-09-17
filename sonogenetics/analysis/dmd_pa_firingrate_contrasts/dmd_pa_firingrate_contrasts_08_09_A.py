import pandas as pd
import numpy as np
import utils
from sonogenetics.analysis.lib.data_io import get_dataio
from sonogenetics.analysis.lib.analysis_tools import detect_preferred_electrode
from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis
from utils import load_obj, make_figure
from typing import Dict
from sonogenetics.analysis.lib.bootstrap import BootstrapOutput

sessions = [
    # '2026-09-08 mouse c57 750 Mekano6 A',
    # '2026-09-08 mouse c57 750 Mekano6 B',
    '2026-09-08 mouse c57 750 Mekano6 C',
    '2026-09-09 mouse c57 758 Mekano6 A',
    '2026-09-09 mouse c57 758 Mekano6 C',
]

def main():
    # Load session data
    for sid in sessions:

        data_io = get_dataio()
        data_io.load_session(sid, load_pickle=True)

        rid = None
        for r in data_io.recording_ids:
            if 'pa_dmd_timing_full_field' in r and 'CNQX' not in r:
                rid = r
                break

        assert rid is not None
        print(f'\tselected: {rid}')

        # Load typing data
        load_name = data_io.datadir / 'response_types' / f'{sid}_cell_response_types.csv'
        if not load_name.exists():
            print('Make sure to run print_nr_response_clusters before....')
            return
        type_df = pd.read_csv(load_name, index_col=0, header=0)

        # Load cell response statistics
        load_name = data_io.datadir / f'{data_io.session_id}_cells.csv'
        cells_df = pd.read_csv(load_name, header=[0, 1], index_col=0)

        # Load preferred electrodes dict
        pref_ec = detect_preferred_electrode(data_io, cells_df)

        # Print response types
        print(f'response types:')
        for k, v in type_df.value_counts('response_type').items():
            print(f'\t{k}: {v}')

        # Print recording IDS
        print(f'\nrecording ids:')
        for r in data_io.recording_ids:
            print(f'\t-{r}')

        # Plot graphs
        ref_rec = rid

        for cid in data_io.cluster_ids:
            ec = pref_ec[ref_rec]
            rtype = type_df.loc[cid, 'response_type']
            protocols = list(ec.keys())
            assert len(protocols) == 1

            if cid not in ec[protocols[0]].index.values:
                continue
            ec = ec[protocols[0]].loc[cid, 'ec']

            # Load response data
            cluster_data: Dict[str, BootstrapOutput] = load_obj(dataset_dir / 'bootstrapped' / f'bootstrap_{cid}.pkl')
            if cluster_data is None:
                continue

            # Extract data for the 3 conditions
            # 1. PA response (pa_intensity_test)
            if pd.isna(ec):
                has_pa = False
            else:
                has_pa = True
                pa_trials = data_io.train_df.query(f'rec_id == "{rid}" and '
                                                   f'electrode == {ec} and '
                                                   f'has_laser == 1 and '
                                                   f'has_dmd == 0 and '
                                                   f'laser_burst_duration == 100'
                                                   )
                max_pwr = pa_trials.laser_power.max()
                pa_trials = pa_trials.query(f'laser_power == {max_pwr}')
                max_prr = pa_trials.laser_pulse_repetition_rate.max()
                pa_trials = pa_trials.query(f'laser_pulse_repetition_rate == {max_prr}')
                assert len(pa_trials) == 1
                train_id = pa_trials.index.values[0]

                if cluster_data[train_id] is None:
                    has_pa = False
                else:
                    pa_bins = cluster_data[train_id].get('bins')
                    pa_fr = cluster_data[train_id].get('firing_rate')
                    pa_bd = pa_trials.laser_burst_duration.values[0]

            # 2. DMD response (DMD full field intensities
            dmd_trials = data_io.train_df.query(f'rec_id == "{rid}" and '
                                               f'has_laser == 0 and '
                                               f'has_dmd == 1')
            max_i = dmd_trials.dmd_intensity.max()
            dmd_trials = dmd_trials.query(f'dmd_intensity == {max_i}')
            assert len(dmd_trials) == 3
            train_id = dmd_trials.index.values[0]

            if cluster_data[train_id] is None:
                has_dmd = False
            else:
                has_dmd = True
                dmd_bins = cluster_data[train_id].get('bins')
                dmd_fr = cluster_data[train_id].get('firing_rate')
                dmd_bd = dmd_trials.dmd_burst_duration.values[0]


            # 3. Dual responses (pa_dmd_timing)
            if has_pa:
                dual_trials = data_io.train_df.query(f'rec_id == "{rid}" and '
                                                   f'electrode == {ec} and '
                                                   f'has_laser == 1 and '
                                                   f'has_dmd == 1 and '
                                                    f'laser_onset_delay == 40 and '
                                                    f'dmd_onset_delay == 0 and '
                                                    f'laser_burst_duration == 100 and '
                                                     f'laser_power == 5000'
                                                     )
                assert len(dual_trials) == 1
                train_id = dual_trials.index.values[0]

                if cluster_data[train_id] is None:
                    continue
                dual_bins = cluster_data[train_id].get('bins')
                dual_fr = cluster_data[train_id].get('firing_rate')
                dual_dmd_bd = dual_trials.dmd_burst_duration.values[0]
                dual_pa_bd = dual_trials.laser_burst_duration.values[0]
                dual_dmd_delay = dual_trials.dmd_onset_delay.values[0]
                dual_pa_delay = dual_trials.laser_onset_delay.values[0]


            # Make figure
            if not has_pa:
                pa_fr = np.array([0])
                dual_fr = np.array([0])

            if not has_dmd:
                dmd_fr = np.array([0])

            y_max = np.max([np.max(pa_fr), np.max(dmd_fr), np.max(dual_fr)])
            y_max += 0.1 * y_max

            fig_x_domains ={
                    1: [[0.1, 0.95]],
            }
            fig_y_domains = {
                    1: [[0.1, 0.95], ],
            }
            fig = make_figure(
                width=1,
                height=1.5,
                x_domains=fig_x_domains,
                y_domains=fig_y_domains,
                xticks=[], yticks=[],
                subplot_titles={1: [f'{rtype}']}
            )

            # Plot PA data
            if has_pa:
                fig.add_scatter(x=pa_bins, y=pa_fr, mode='lines', name=f'PA-response', showlegend=True, line=dict(color='red', width=1.5),)
                fig.add_scatter(x=[0, 0, pa_bd, pa_bd, 0], y=[0, 0.3*y_max, 0.3*y_max, 0, 0], mode='lines',
                                line=dict(color='red', width=1), fill='toself', showlegend=True, fillcolor='rgba(255, 0, 0, 0.1)',
                                name='PA-stim')

            # Plot DMD data
            if has_dmd:
                fig.add_scatter(x=dmd_bins, y=dmd_fr, mode='lines', name='DMD-response', showlegend=True, line=dict(color='blue', width=1.5)    )
                fig.add_scatter( x=[0, 0, dmd_bd, dmd_bd, 0], y=[0.3*y_max, 0.6*y_max, 0.6*y_max, 0.3*y_max, 0.3*y_max],
                    mode='lines', line=dict(color='blue', width=1), fill='toself', showlegend=True, fillcolor='rgba(0, 0, 255, 0.1)',
                                 name='DMD-stim')

            # Plot dual data
            if has_pa:
                fig.add_scatter(x=dual_bins, y=dual_fr, mode='lines', name='DUAL-response', showlegend=True, line=dict(color='purple', width=1.5))
                fig.add_scatter(x=[dual_dmd_delay, dual_dmd_delay, dual_dmd_delay+dual_dmd_bd, dual_dmd_delay+dual_dmd_bd, dual_dmd_delay],
                                y=[0.6*y_max, y_max, y_max, 0.6*y_max, 0.6 * y_max], mode='lines', line=dict(color='purple', width=1), fill='toself',
                                showlegend=False, fillcolor='rgba(0, 0, 255, 0.1)')
                fig.add_scatter(x=[dual_pa_delay, dual_pa_delay, dual_pa_delay+dual_pa_bd, dual_pa_delay+dual_pa_bd, dual_pa_delay],
                                y=[0.6*y_max, y_max, y_max, 0.6*y_max, 0.6*y_max], mode='lines', line=dict(color='purple', width=1), fill='toself',
                                fillcolor='rgba(255, 0, 0, 0.1)',
                                showlegend=False,)


            fig.update_xaxes(
                tickvals=np.arange(-200, 500, 100),
                title_text='Time [ms]'
            )
            fig.update_yaxes(
                range=[0, y_max],
                title_text='Fr'
            )

            savename = figure_dir_analysis / 'dmd_pa_firing_contrasts' / sid / f'{cid}'
            utils.save_fig(fig, savename, display=False, backend='yes')


if __name__ == '__main__':
    main()