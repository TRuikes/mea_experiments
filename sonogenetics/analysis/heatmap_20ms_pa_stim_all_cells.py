from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis
from sonogenetics.analysis.lib.data_io import DataIO
import pandas as pd
from sonogenetics.analysis.lib.analysis_tools import detect_preferred_electrode
from utils import  make_figure, save_fig, load_obj
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sonogenetics.analysis.lib.poisson_rate_estimation import PoissonOutput
from typing import Dict

MIN_D = 3000
LASER_POWER = 5000
LASER_PRR = 5000
LASER_BURST_DURATION = 20

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


data_list = (
    # ('2026-05-19 mouse c57 Audrey A', '260519_A_005_noblocker_pa_prr_series'),
    # ('2026-05-19 mouse c57 Audrey A', '260519_A_009_acet_lap4_pa_prr_series'),
    # ('2026-05-20 mouse c57 Audrey A', '260520_A_005_noblocker_pa_prr_series'),
    # ('2026-05-20 mouse c57 Audrey A', '260520_A_010_acet_lap4_pa_prr_series'),
    # ('2026-02-11 mouse c57 565 eMSCL A', 'rec_2_pilot_021126'),
    # ('2026-02-16 mouse c57 566 eMSCL A', 'rec_1_26-02-16_A_pilot021626_noblocker'),
    # ('2026-02-19 mouse c57 5713 Mekano6 A', 'rec_2_pa_dose_sequence_1'),
    # ('2026-03-25 mouse c57 617 Mekano6 B', 'rec_1_B_20260325_pa_intensity_test'),
    # ('2026-05-13 mouse c57 615 Mekano6 A', 'rec_2_2026-05-13_mouse_615_A_pa_intensity_test'),
    # ('2026-06-12 mouse c57 649 Mekano6 C', 'rec_2_C_20260612_pa_intensity_test'),
    # ('2026-06-12 mouse c57 649 Mekano6 D', 'rec_2_D_20260612_pa_intensity_test'),
    # ('2026-06-16 mouse c57 645 Mekano6 B', 'rec_2_B_20260616_pa_intensity_test'),
    # ('2026-06-30 rat LE 803 Mekano6 B', 'rec_2_B_20260630_pa_intensity_test'),
    # ('2026-07-01 mouse c57 653 NoVirus C', 'rec_2_C_20260701_pa_intensity_test'),
    # ('2026-07-02 mouse c57 650 Mekano6 A', 'rec_2_A_20260702_pa_intensity_test'),
    ('2026-07-08 rat LE 3322 Mekano6 A', 'rec_2_A_20260708_pa_dmd_timing_full_field'),
    ('2026-07-08 rat LE 3322 Mekano6 A', 'rec_3_A_20260708_pa_dmd_timing_full_field_RSCPP_CNQX'),
    ('2026-07-08 rat LE 3322 Mekano6 B', 'rec_2_B_20260708_pa_dmd_timing_full_field'),
    ('2026-07-08 rat LE 3322 Mekano6 B', "rec_3_B_20260708_pa_dmd_timing_full_field_RSCPP_CNQX"),
    ('2026-07-09 rat LE 0353 Mekano6 A', 'rec_2_A_20260709_pa_dmd_timing_full_field'),
    ('2026-07-09 rat LE 0353 Mekano6 A', 'rec_4_A_20260709_pa_dmd_timing_full_field_RSCPP_CNQX'),
    ('2026-07-09 rat LE 0353 Mekano6 B', 'rec_2_B_20260709_pa_dmd_timing_full_field'),
    ('2026-07-09 rat LE 0353 Mekano6 B', 'rec_3_B_20260709_pa_dmd_timing_full_field_RSCPP_CNQX'),
)


def main():
    # Load dataset + dump as pickle to speedup future data loading
    data_io = DataIO(dataset_dir)

    for session_id, rec_id in data_list:
        data_io.load_session(session_id, load_pickle=False, load_waveforms=False)
        data_io.dump_as_pickle()

        loadname = dataset_dir / f'{data_io.session_id}_cells.csv'
        cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
        pref_ec = detect_preferred_electrode(data_io, cells_df)

        if session_id == '2026-02-11 mouse c57 565 eMSCL A':
            laser_ppr_to_use = 4000
        elif session_id in ['2026-02-16 mouse c57 566 eMSCL A', '2026-02-19 mouse c57 5713 Mekano6 A']:
            laser_ppr_to_use = 6000

        else:
            laser_ppr_to_use = LASER_PRR

        if session_id == '2026-02-16 mouse c57 566 eMSCL A':
            laser_power_to_use = 6000
        else:
            laser_power_to_use = LASER_POWER

        train_df = data_io.train_df.query(
            f'rec_id ==  "{rec_id}" and '
            f'laser_burst_duration == {LASER_BURST_DURATION} and '
            f'laser_pulse_repetition_rate == {laser_ppr_to_use} and '
            f'laser_power == {laser_power_to_use} and '
            f'has_dmd == False'
        )

        frates = []
        latencies = []
        labels = []

        stim_durations = train_df.laser_burst_duration.unique()
        assert len(stim_durations) == 1
        stim_duration = stim_durations[0]

        for tid, tinfo in train_df.iterrows():
            burst_onsets = data_io.burst_df.query(f'train_id == "{tid}"').laser_burst_onset

            for cid in data_io.cluster_ids:

                if pref_ec[rec_id][tinfo.protocol_name].loc[cid].ec != tinfo.electrode:
                    continue

                # if cells_df.loc[cid][tid]['laser_distance'] > MIN_D:
                #     continue

                if not cells_df.loc[cid][tid, 'is_excited'] and not cells_df.loc[cid][tid, 'is_inhibited']:
                    continue

                # if cells_df.loc[cid][tid, 'is_excited'] or cells_df.loc[cid][tid, 'is_inhibited']:
                #     continue

                if pd.isna(cells_df.loc[cid][tid, 'is_excited']) and pd.isna(cells_df.loc[cid][tid, 'is_inhibited']):
                    continue

                cluster_data: Dict[str, PoissonOutput] = load_obj(dataset_dir / 'bootstrapped' / f'bootstrap_{cid}.pkl')

                if cluster_data[tid] is None:
                    continue

                mean_fr = cluster_data[tid].firing_rate

                bin_centres = cluster_data[tid].bins
                baseline_mask = (bin_centres >= baseline_t0) & (bin_centres < baseline_t1)

                if cells_df.loc[cid][tid, 'is_inhibited'] and cluster_data[tid].baseline_firing_rate_mean * 1e3 > 20:
                    frates.append(mean_fr)
                    labels.append(cid)
                    latencies.append(cells_df.loc[cid][tid, 'inhibition_start'])
                    # print('in ', cid, tid, f'{latencies[-1]:.2f}',
                    #       f'{cluster_data[tid].baseline_firing_rate_mean * 1e3:.0f}')

                elif cells_df.loc[cid][tid, 'is_excited']:
                        frates.append(mean_fr)
                        labels.append(cid)
                        latencies.append(cells_df.loc[cid][tid, 'excitation_start'])
                        # print('ex ', cid, tid, f'{latencies[-1]:.2f}', f'{cluster_data[tid].baseline_firing_rate_mean * 1e3:.0f}')

                else:
                    if cells_df.loc[cid][tid, 'is_inhibited'] and cluster_data[tid].baseline_firing_rate_mean * 1e3 <= 20:
                        continue

                    raise ValueError('test')
                    # latencies.append(0)
                    # print('non', cid, tid, f'{0}', f'{cluster_data[tid].baseline_firing_rate_mean * 1e3:.0f}')

        cell_fr = np.array(frates)
        cell_fr = cell_fr / (binwidth / 1000)

        # -----------------------------
        # Z-score to baseline window
        # -----------------------------
        if cell_fr.shape[0] == 0:
            # raise ValueError('no cell found')
            print('no cells found')
            continue
        baseline = cell_fr[:, baseline_mask]

        baseline_mean = baseline.mean(axis=1, keepdims=True)
        baseline_std = baseline.std(axis=1, keepdims=True)

        # baseline_std[baseline_std == 0] = 1  # avoid divide-by-zero
        idx = np.where(baseline != 0)[0]
        cell_fr[idx] = (cell_fr[idx] - baseline_mean[idx]) / baseline_std[idx]

        # Optional clipping
        # cell_fr = np.clip(cell_fr, -max_z, max_z)

        # Sort based on latency
        sort_idx = np.argsort(latencies)
        cell_fr = cell_fr[sort_idx, :]
        labels = [labels[i] for i in sort_idx]
        latencies = np.array([latencies[i] for i in sort_idx])

        # 1. Clip the data to your strict boundaries
        clipped_fr = np.clip(cell_fr, zmin, zmax)

        # Create a copy to hold our normalized data
        norm_fr = np.zeros_like(clipped_fr)

        # 2. Process row by row independently
        for i in range(clipped_fr.shape[0]):
            row = clipped_fr[i, :]

            # Separate masks for positive and negative values
            pos_mask = row > 0
            neg_mask = row < 0

            # Normalize positive values to [0, 1] relative to zmax
            if np.any(pos_mask):
                # We divide by zmax to map the theoretical max (12) to 1.0
                norm_fr[i, pos_mask] = row[pos_mask] / zmax

            # Normalize negative values to [-1, 0] relative to zmin
            if np.any(neg_mask):
                # We divide by abs(zmin) to map the theoretical min (-2) to -1.0
                norm_fr[i, neg_mask] = row[neg_mask] / abs(zmin)

        # 3. Apply Non-Linear Scaling
        # np.sign preserves the direction (+ or -), while np.abs() ** exponent bends the curve
        exponent = 2.0  # Powers > 1 suppress values near 0. Adjust this to tune the effect!
        nonlinear_fr = np.sign(norm_fr) * (np.abs(norm_fr) ** exponent)

        # -----------------------------
        # Plot heatmap
        # -----------------------------
        fig = make_figure(
            height=2,
            x_domains={1: [[0.1, 0.9]]},
            y_domains={1: [[0.1, 0.9]]},
            subplot_titles={1: [f'PA power: {laser_power_to_use:.0f}, PA PRR: {laser_ppr_to_use/1e3:.1f} kHz, PA duration: {stim_duration:.0f} ms']}
        )


        fig.add_heatmap(
            z=nonlinear_fr,
            x=bin_centres,
            y=np.arange(cell_fr.shape[0]),
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(
                lenmode='fraction',
                len=0.4,
                title=dict(
                    text='Z-scored FR',
                    side='right',
                    font=dict(size=10, color='black')
                    # textangle=-90
                ),
                tickmode='array',
                tickvals=np.arange(-1, 2, 1),
                tickfont=dict(size=8, color='black'),
                thickness=15,
                x=0.98
            )
        )

        fig.add_scatter(
            x=[0, 0, stim_duration, stim_duration],
            y=[-1, cell_fr.shape[0]+1, cell_fr.shape[0]+1, -1],
            mode='lines',
            line=dict(color='red', width=0.0001),
            fill='toself',
            fillcolor='rgba(179, 157, 219, 0.4)',
            showlegend=False,
        )
        #
        # fig.add_scatter(
        #     x=bin_centres,
        #     y=np.ones_like(bin_centres) * nonzero_idx,
        #     mode='lines', line=dict(color='black', width=2),
        #     showlegend=False
        # )

        fig.update_xaxes(
            tickvals=np.arange(-200, 301, 50),
            title_text='time [ms]'
        )

        fig.update_yaxes(
            title_text = 'cell #',
            range=[-0.5, cell_fr.shape[0]-0.5],
            # tickvals=np.arange(0, len(labels)),
            # ticktext=labels
        )

        savename = figure_dir_analysis / 'heatmap_all_cells' / f'{session_id}_{rec_id}'
        save_fig(fig=fig, savename=savename, display=False)




if __name__ == '__main__':
    main()

