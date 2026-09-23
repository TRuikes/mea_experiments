import pandas as pd
import utils
from sonogenetics.analysis.lib.data_io import get_dataio
from sonogenetics.analysis.lib.analysis_tools import detect_preferred_electrode
from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis
from utils import load_obj, make_figure
from typing import Dict
from sonogenetics.analysis.lib.bootstrap import BootstrapOutput
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load session data
    sid = '2026-07-02 mouse c57 650 Mekano6 A'
    data_io = get_dataio()
    data_io.load_session(sid, load_pickle=False)


    # Load cell response statistics
    load_name = data_io.datadir / f'{data_io.session_id}_cells.csv'
    cells_df = pd.read_csv(load_name, header=[0, 1], index_col=0)

    # Load preferred electrodes dict
    pref_ec = detect_preferred_electrode(data_io, cells_df)

    # Print recording IDS
    print(f'\nrecording ids:')
    for r in data_io.recording_ids:
        print(f'\t-{r}')

    # Plot graphs
    ref_rec = 'rec_2_A_20260702_pa_intensity_test'

    for cid in data_io.cluster_ids:
        ec = pref_ec[ref_rec]
        protocols = list(ec.keys())
        assert len(protocols) == 1
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
            pa_trials = data_io.train_df.query(f'rec_id == "rec_2_A_20260702_pa_intensity_test" and '
                                               f'electrode == {ec} and '
                                               f'has_laser == 1 and '
                                               f'has_dmd == 0')
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
        dmd_trials = data_io.train_df.query(f'rec_id == "rec_5_A_20260702_dmd_full_field_intensities" and '
                                           f'has_laser == 0 and '
                                           f'has_dmd == 1')
        max_i = dmd_trials.dmd_intensity.max()
        dmd_trials = dmd_trials.query(f'dmd_intensity == {max_i}')
        assert len(dmd_trials) == 1
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
            dual_trials = data_io.train_df.query(f'rec_id == "rec_4_A_20260702_pa_dmd_timing" and '
                                               f'electrode == {ec} and '
                                               f'has_laser == 1 and '
                                               f'has_dmd == 1 and '
                                                f'laser_onset_delay == 40 and '
                                                f'dmd_onset_delay == 0 and '
                                                f'laser_burst_duration == 20'
                                                 )
            assert len(dual_trials) == 1
            train_id = dual_trials.index.values[0]

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
        # fig = make_figure(
        #     width=1,
        #     height=1.5,
        #     x_domains=fig_x_domains,
        #     y_domains=fig_y_domains,
        #     xticks=[], yticks=[],
        #     subplot_titles={1: [f'']}
        # )
        #
        # # Plot PA data
        # if has_pa:
        #     fig.add_scatter(x=pa_bins, y=pa_fr, mode='lines', name=f'PA-response', showlegend=True, line=dict(color='red', width=1.5),)
        #     fig.add_scatter(x=[0, 0, pa_bd, pa_bd, 0], y=[0, 0.3*y_max, 0.3*y_max, 0, 0], mode='lines',
        #                     line=dict(color='red', width=1), fill='toself', showlegend=True, fillcolor='rgba(255, 0, 0, 0.1)',
        #                     name='PA-stim')
        #
        # # Plot DMD data
        # if has_dmd:
        #     fig.add_scatter(x=dmd_bins, y=dmd_fr, mode='lines', name='DMD-response', showlegend=True, line=dict(color='blue', width=1.5)    )
        #     fig.add_scatter( x=[0, 0, dmd_bd, dmd_bd, 0], y=[0.3*y_max, 0.6*y_max, 0.6*y_max, 0.3*y_max, 0.3*y_max],
        #         mode='lines', line=dict(color='blue', width=1), fill='toself', showlegend=True, fillcolor='rgba(0, 0, 255, 0.1)',
        #                      name='DMD-stim')
        #
        # # Plot dual data
        # if has_pa:
        #     fig.add_scatter(x=dual_bins, y=dual_fr, mode='lines', name='DUAL-response', showlegend=True, line=dict(color='purple', width=1.5))
        #     fig.add_scatter(x=[dual_dmd_delay, dual_dmd_delay, dual_dmd_delay+dual_dmd_bd, dual_dmd_delay+dual_dmd_bd, dual_dmd_delay],
        #                     y=[0.6*y_max, y_max, y_max, 0.6*y_max, 0.6 * y_max], mode='lines', line=dict(color='purple', width=1), fill='toself',
        #                     showlegend=False, fillcolor='rgba(0, 0, 255, 0.1)')
        #     fig.add_scatter(x=[dual_pa_delay, dual_pa_delay, dual_pa_delay+dual_pa_bd, dual_pa_delay+dual_pa_bd, dual_pa_delay],
        #                     y=[0.6*y_max, y_max, y_max, 0.6*y_max, 0.6*y_max], mode='lines', line=dict(color='purple', width=1), fill='toself',
        #                     fillcolor='rgba(255, 0, 0, 0.1)',
        #                     showlegend=False,)
        #
        #
        # fig.update_xaxes(
        #     tickvals=np.arange(-200, 500, 100),
        #     title_text='Time [ms]'
        # )
        # fig.update_yaxes(
        #     range=[0, y_max],
        #     title_text='Fr'
        # )
        #
        # savename = figure_dir_analysis / 'dmd_pa_firing_contrasts' / sid / f'{cid}'
        # utils.save_fig(fig, savename, display=False)



        # Create a figure with the specified size
        fig, ax = plt.subplots(figsize=(15, 10))  # Width: 4 inches, Height: 6 inches (1:1.5 ratio)

        # Plot PA data
        if has_pa:
            ax.plot(pa_bins, pa_fr, color='red', linewidth=1.5, label='PA-response')
            ax.fill([0, 0, pa_bd, pa_bd, 0], [0, 0.3 * y_max, 0.3 * y_max, 0, 0], color='red', alpha=0.1,
                    label='PA-stim')

        # Plot DMD data
        if has_dmd:
            ax.plot(dmd_bins, dmd_fr, color='blue', linewidth=1.5, label='DMD-response')
            ax.fill([0, 0, dmd_bd, dmd_bd, 0], [0.3 * y_max, 0.6 * y_max, 0.6 * y_max, 0.3 * y_max, 0.3 * y_max],
                    color='blue', alpha=0.1, label='DMD-stim')

        # Plot dual data
        if has_pa and has_dmd:  # Assuming dual data requires both PA and DMD
            ax.plot(dual_bins, dual_fr, color='purple', linewidth=1.5, label='DUAL-response')
            ax.fill([dual_dmd_delay, dual_dmd_delay, dual_dmd_delay + dual_dmd_bd, dual_dmd_delay + dual_dmd_bd,
                     dual_dmd_delay],
                    [0.6 * y_max, y_max, y_max, 0.6 * y_max, 0.6 * y_max], color='blue', alpha=0.1)
            ax.fill(
                [dual_pa_delay, dual_pa_delay, dual_pa_delay + dual_pa_bd, dual_pa_delay + dual_pa_bd, dual_pa_delay],
                [0.6 * y_max, y_max, y_max, 0.6 * y_max, 0.6 * y_max], color='red', alpha=0.1)

        # Set x and y axis properties
        ax.set_xticks(np.arange(-200, 500, 100))
        ax.set_xlabel('Time [ms]')
        ax.set_xlim(-100, 200)
        ax.set_ylim(0, y_max)
        ax.set_ylabel('Fr')

        # Add legend
        ax.legend()

        # Save the figure
        savename = figure_dir_analysis / 'dmd_pa_firing_contrasts' / sid / f'{cid}.png'
        plt.savefig(savename, dpi=200, bbox_inches='tight', format='png')
        plt.close()


if __name__ == '__main__':
    main()