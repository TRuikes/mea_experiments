import pandas as pd
import numpy as np
from mypy.main import process_options

import utils
from sonogenetics.analysis.lib.data_io import get_dataio, DataIO
from sonogenetics.analysis.lib.analysis_tools import detect_preferred_electrode
from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis
from utils import load_obj, make_figure
from typing import Dict
from sonogenetics.analysis.lib.bootstrap import BootstrapOutput
from sonogenetics.analysis.data_list import data_list
import matplotlib.pyplot as plt
from tqdm import tqdm


session_ids = [
    # '2026-07-08 rat LE 3322 Mekano6 A',
    # '2026-07-08 rat LE 3322 Mekano6 B',  # doesnt work?
    # '2026-07-09 rat LE 0353 Mekano6 A',
    # '2026-07-09 rat LE 0353 Mekano6 B',
    # '2026-09-08 mouse c57 750 Mekano6 A',
    # '2026-09-08 mouse c57 750 Mekano6 B',
    # '2026-09-08 mouse c57 750 Mekano6 C',
    # '2026-09-09 mouse c57 758 Mekano6 A',
    # '2026-09-09 mouse c57 758 Mekano6 C',  # doesn't work
    # '2026-09-15 mouse c57 752 Mekano6 A',
    # '2026-09-15 mouse c57 752 Mekano6 B',
    # '2026-09-15 mouse c57 752 Mekano6 C',

    '2026-08-25 mouse c57 754 eMSCL A',
    '2026-08-25 mouse c57 754 eMSCL B',
    '2026-08-25 mouse c57 754 eMSCL C',
    '2026-08-26 mouse c57 755 eMSCL A',
    '2026-08-26 mouse c57 755 eMSCL B',
    '2026-08-26 mouse c57 755 eMSCL C',
]

laser_bust_duration = 100


def main():
    # Load session data
    data_io = get_dataio()

    for sid in session_ids:
        data_io.load_session(sid, load_pickle=False)



        # Print recording IDS
        print(f'\n{sid} recording ids:')
        for r in data_io.recording_ids:
            if 'pa_dmd_timing' in r:
                print(f'\t-{r}')

                plot(data_io=data_io, rec_id=r)

def plot(data_io: DataIO, rec_id: str):
    # Load cell response statistics
    load_name = data_io.datadir / f'{data_io.session_id}_cells.csv'
    cells_df = pd.read_csv(load_name, header=[0, 1], index_col=0)

    # Load preferred electrodes dict
    pref_ec = detect_preferred_electrode(data_io, cells_df)

    # Define the output directory and full path
    output_dir = (
            figure_dir_analysis / "dmd_pa_firing_contrasts" / data_io.session_id / rec_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
    print(f'saving data in: {output_dir}')

    for cid in tqdm(data_io.cluster_ids):
        ec = pref_ec[rec_id]
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
            continue
        else:
            has_pa = True
            pa_trials = data_io.train_df.query(f'rec_id == "{rec_id}" and '
                                               f'electrode == {ec} and '
                                               f'has_laser == 1 and '
                                               f'laser_burst_duration == {laser_bust_duration} and '
                                               f'has_dmd == 0'
                                               )
            max_pwr = pa_trials.laser_power.max()
            pa_trials = pa_trials.query(f'laser_power == {max_pwr}')
            max_prr = pa_trials.laser_pulse_repetition_rate.max()
            pa_trials = pa_trials.query(f'laser_pulse_repetition_rate == {max_prr}')
            assert len(pa_trials) == 1
            train_id = pa_trials.index.values[0]

            if cluster_data[train_id] is None:
                has_pa = False
                continue
            else:
                pa_bins = cluster_data[train_id].get('bins')
                pa_fr = cluster_data[train_id].get('firing_rate')
                pa_bd = pa_trials.laser_burst_duration.values[0]

        # 2. DMD response (DMD full field intensities
        dmd_trials = data_io.train_df.query(f'rec_id == "{rec_id}" and '
                                           f'has_laser == 0 and '
                                           f'has_dmd == 1')
        max_i = dmd_trials.dmd_intensity.max()
        dmd_trials = dmd_trials.query(f'dmd_intensity == {max_i}')
        # assert len(dmd_trials) == 1 or len(dmd_trials)
        if len(dmd_trials) > 1:
            assert len(dmd_trials) == len(dmd_trials.electrode.unique())

        train_id = dmd_trials.index.values[0]

        if cluster_data[train_id] is None:
            has_dmd = False
            continue
        else:
            has_dmd = True
            dmd_bins = cluster_data[train_id].get('bins')
            dmd_fr = cluster_data[train_id].get('firing_rate')
            dmd_bd = dmd_trials.dmd_burst_duration.values[0]


        # 3. Dual responses (pa_dmd_timing)
        if has_pa:
            dual_trials = data_io.train_df.query(f'rec_id == "{rec_id}" and '
                                               f'electrode == {ec} and '
                                               f'has_laser == 1 and '
                                               f'has_dmd == 1 and '
                                                 f'laser_power == 4000 and '
                                                f'laser_onset_delay == 40 and '
                                                f'dmd_onset_delay == 0 and '
                                                f'laser_burst_duration == {laser_bust_duration}'
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

        # fig_x_domains ={
        #         1: [[0.1, 0.95]],
        # }
        # fig_y_domains = {
        #         1: [[0.1, 0.95], ],
        # }

        #
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
        # savename = figure_dir_analysis / 'dmd_pa_firing_contrasts' / data_io.session_id / f'{rec_id}-{cid}'
        # utils.save_fig(fig, savename, display=False)



        # Create figure and axis (converting width/height to inches)
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot PA data
        ax.plot(
            pa_bins, pa_fr, color="red", linewidth=1.5, label="PA-response"
        )
        ax.fill(
            [0, 0, pa_bd, pa_bd, 0],
            [0, 0.3 * y_max, 0.3 * y_max, 0, 0],
            facecolor=(1, 0, 0, 0.1),
            edgecolor="red",
            linewidth=1,
            label="PA-stim",
        )

        # Plot DMD data

        ax.plot(
            dmd_bins, dmd_fr, color="blue", linewidth=1.5, label="DMD-response"
        )
        ax.fill(
            [0, 0, dmd_bd, dmd_bd, 0],
            [0.3 * y_max, 0.6 * y_max, 0.6 * y_max, 0.3 * y_max, 0.3 * y_max],
            facecolor=(0, 0, 1, 0.1),
            edgecolor="blue",
            linewidth=1,
            label="DMD-stim",
        )

        # Plot dual data
        ax.plot(
            dual_bins,
            dual_fr,
            color="purple",
            linewidth=1.5,
            label="DUAL-response",
        )
        ax.fill(
            [
                dual_dmd_delay,
                dual_dmd_delay,
                dual_dmd_delay + dual_dmd_bd,
                dual_dmd_delay + dual_dmd_bd,
                dual_dmd_delay,
            ],
            [0.6 * y_max, y_max, y_max, 0.6 * y_max, 0.6 * y_max],
            facecolor=(0, 0, 1, 0.1),
            edgecolor="purple",
            linewidth=1,
        )
        ax.fill(
            [
                dual_pa_delay,
                dual_pa_delay,
                dual_pa_delay + dual_pa_bd,
                dual_pa_delay + dual_pa_bd,
                dual_pa_delay,
            ],
            [0.6 * y_max, y_max, y_max, 0.6 * y_max, 0.6 * y_max],
            facecolor=(1, 0, 0, 0.1),
            edgecolor="purple",
            linewidth=1,
        )

        # Axis formatting
        ax.set_xticks(np.arange(-200, 500, 100))
        ax.set_xlabel("Time [ms]")
        ax.set_xlim(-100, 200)
        ax.set_ylim(0, y_max)
        ax.set_ylabel("Fr")
        ax.legend()



        savename = output_dir / f'laser_bd_{laser_bust_duration:.0f}' / f"{rec_id}-{cid}.png"  # Add file extension (e.g., .png or .pdf)
        if not savename.parent.exists():
            savename.parent.mkdir(parents=True)

        # Save figure using Matplotlib
        fig.savefig(savename, dpi=300, bbox_inches="tight")
        # Clean up memory
        plt.close(fig)

    # utils.save_fig(fig, savename, display=False)

if __name__ == '__main__':
    main()