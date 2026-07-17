import utils
from sonogenetics.analysis.lib.bootstrap import BootstrapOutput
from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis
from sonogenetics.analysis.lib.data_io import DataIO
import pandas as pd
from sonogenetics.analysis.lib.analysis_tools import detect_preferred_electrode
from utils import  make_figure, save_fig, load_obj
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sonogenetics.analysis.lib.poisson_rate_estimation import PoissonOutput
from typing import Dict
from sonogenetics.project_colors import ProjectColors


MIN_D = 3000
LASER_POWER = 5000
LASER_PRR = 5000
LASER_BURST_DURATION = 100

t_pre = 30
t_after = 200  # to focus on 0-50 ms
stepsize = 1
binwidth = 5

smooth_sigma = 3
baseline_t0 = -120
baseline_t1 = 0

# Make an asymmetric colorscale
zmin = -2
zmax = 12


data_list = {
    # '2026-07-08 rat LE 3322 Mekano6 A': ['rec_2_A_20260708_pa_dmd_timing_full_field',  'rec_3_A_20260708_pa_dmd_timing_full_field_RSCPP_CNQX'],
    '2026-07-08 rat LE 3322 Mekano6 B': ['rec_2_B_20260708_pa_dmd_timing_full_field',  "rec_3_B_20260708_pa_dmd_timing_full_field_RSCPP_CNQX"],
    '2026-07-09 rat LE 0353 Mekano6 A': ['rec_2_A_20260709_pa_dmd_timing_full_field', 'rec_4_A_20260709_pa_dmd_timing_full_field_RSCPP_CNQX'],
    '2026-07-09 rat LE 0353 Mekano6 B': ['rec_2_B_20260709_pa_dmd_timing_full_field', 'rec_3_B_20260709_pa_dmd_timing_full_field_RSCPP_CNQX'],
}

trial_specs = {
    'D': dict(has_dmd=True, has_laser=False),
    'DL_0': dict(has_dmd=True, has_laser=True, laser_power=LASER_POWER,
                       laser_burst_duration=LASER_BURST_DURATION, laser_pulse_repetition_rate=LASER_PRR,
                 laser_onset_delay=0),
    'DL_40': dict(has_dmd=True, has_laser=True, laser_power=LASER_POWER,
                 laser_burst_duration=LASER_BURST_DURATION, laser_pulse_repetition_rate=LASER_PRR,
                 laser_onset_delay=40
                  ),
    'DL_60': dict(has_dmd=True, has_laser=True, laser_power=LASER_POWER,
                 laser_burst_duration=LASER_BURST_DURATION, laser_pulse_repetition_rate=LASER_PRR,
                 laser_onset_delay=60),
    'L': dict(has_dmd=False, has_laser=True, laser_power=LASER_POWER,
                       laser_burst_duration=LASER_BURST_DURATION, laser_pulse_repetition_rate=LASER_PRR),
}


def plot_single_cell_firing_rate(cid, ec, data_io, savename):
    cluster_data: Dict[str, BootstrapOutput] = load_obj(
        dataset_dir / 'bootstrapped' / f'bootstrap_{cid}.pkl')

    x_domains = {1: [[0.1, 0.9]],
                 2: [[0.1, 0.9]]}
    y_domains = {1: [[0.55, 0.9]],
                2: [[0.1, 0.45]]}

    clrs = ProjectColors()

    fig = make_figure(
        height=1.5,
        width=1,
        x_domains=x_domains,
        y_domains=y_domains,
        subplot_titles={1: ['WT'],
                        2: ['CX']}
    )

    ym = 0
    title_txt = ''
    for i, rec_id in enumerate(data_list[data_io.session_id]):
        pos = dict(row=i+1, col=1)
        for data_name, data_specs in trial_specs.items():
            tid = data_io.get_tid_by_specs(**data_specs, electrode=ec, rec_id=rec_id)
            if tid is None:
                continue
            data: BootstrapOutput = cluster_data[tid]
            if data is None:
                continue
            x = data.bins
            y = data.firing_rate
            y_neg = data.firing_rate_ci_low
            y_pos = data.firing_rate_ci_high
            if np.max(y_pos) > ym:
                ym = np.max(y_pos)

            clr = clrs.laser_delay(data_name, 1)
            clr_a = clrs.laser_delay(data_name, 0.3)

            fig.add_scatter(x=x, y=y_neg, mode='lines', line=dict(width=0.001, color=clr_a), showlegend=False, **pos,)
            fig.add_scatter(x=x, y=y_pos, mode='lines', line=dict(width=0.001, color=clr_a), showlegend=False,
                            fill='tonexty', fillcolor=clr_a, **pos,)
            fig.add_scatter(x=x, y=y, mode='lines', line=dict(width=2, color=clr), showlegend=False,  **pos,)


    fig.update_xaxes(
        tickvals=np.arange(-200, 200, 100),
        title_text='time [ms]'
    )

    ym = ym + 0.1 * ym
    fig.update_yaxes(
        range=[-0.05*ym, ym]
    )

    save_fig(fig, savename, display=False)



def main():
    data_io = DataIO(dataset_dir)
    for session_id, (rid_WT, rid_CX) in data_list.items():
        data_io.load_session(session_id, load_pickle=True, load_waveforms=False)
        data_io.dump_as_pickle()

        loadname = dataset_dir / f'{data_io.session_id}_cells.csv'
        cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
        pref_ec = detect_preferred_electrode(data_io, cells_df)

        # Select EC during CNQX recording if there is one, otherwise during WT
        clusters_to_plot = pref_ec[rid_CX]['pa_dmd_timing_full_field'].dropna().copy()
        for i, r in pref_ec[rid_WT]['pa_dmd_timing_full_field'].iterrows():
            if i not in clusters_to_plot.index.values and pd.notna(r.ec):
                clusters_to_plot.at[i, 'ec'] = r.ec

        for cid, cinfo in clusters_to_plot.iterrows():
            for rtype, rec_id in zip(['WT', 'CX'], [rid_WT, rid_CX]):
                tid = data_io.get_tid_by_specs(**trial_specs['DL_60'], electrode=cinfo.ec, rec_id=rec_id)

                clusters_to_plot.at[cid, f'tid_{rtype}'] = tid
                if cells_df.loc[cid][tid, 'is_excited'] or cells_df.loc[cid][tid, 'is_inhibited']:
                    is_sig = True
                else:
                    is_sig = False
                clusters_to_plot.at[cid, f'sig_{rtype}'] = is_sig

        to_drop = []
        for i, r in clusters_to_plot.iterrows():
            if r['sig_WT'] and r['sig_CX']:
                clusters_to_plot.at[i, 'lbl'] = '2_both'
            elif not r['sig_WT'] and r['sig_CX']:
                clusters_to_plot.at[i, 'lbl'] = '1_CX'
            elif r['sig_WT'] and not r['sig_CX']:
                clusters_to_plot.at[i, 'lbl'] = '0_WT'
            else:
                to_drop.append(i)

            cluster_data: Dict[str, BootstrapOutput] = load_obj(
                dataset_dir / 'bootstrapped' / f'bootstrap_{i}.pkl')
            if cluster_data[r['tid_WT']] is None or cluster_data[r['tid_CX']] is None:
                to_drop.append(i)


        clusters_to_plot.drop(index=to_drop, inplace=True)
        clusters_to_plot.sort_values(by=['lbl'], inplace=True)

        for cid, cinfo in clusters_to_plot.iterrows():
            savename = figure_dir_analysis / 'firing_rates_delays' / f'{cid}'
            plot_single_cell_firing_rate(cid, cinfo.ec, data_io, savename)







if __name__ == '__main__':
    main()

