import utils
from sonogenetics.analysis.lib.bootstrap import BootstrapOutput
from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis
from sonogenetics.analysis.lib.data_io import DataIO
import pandas as pd
from sonogenetics.analysis.lib.analysis_tools import detect_preferred_electrode
from utils import  make_figure, save_fig, load_obj
import numpy as np
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


data_list = {
    '2026-07-08 rat LE 3322 Mekano6 A': ['rec_2_A_20260708_pa_dmd_timing_full_field',  'rec_3_A_20260708_pa_dmd_timing_full_field_RSCPP_CNQX'],
    '2026-07-08 rat LE 3322 Mekano6 B': ['rec_2_B_20260708_pa_dmd_timing_full_field',  "rec_3_B_20260708_pa_dmd_timing_full_field_RSCPP_CNQX"],
    '2026-07-09 rat LE 0353 Mekano6 A': ['rec_2_A_20260709_pa_dmd_timing_full_field', 'rec_4_A_20260709_pa_dmd_timing_full_field_RSCPP_CNQX'],
    '2026-07-09 rat LE 0353 Mekano6 B': ['rec_2_B_20260709_pa_dmd_timing_full_field', 'rec_3_B_20260709_pa_dmd_timing_full_field_RSCPP_CNQX'],
}


def plot_single_cell_firing_rate(cluster_data, tid_WT, tid_CX, savename):

    x_domains = {1: [[0.1, 0.9]]}
    y_domains = {1: [[0.1, 0.9]]}
    fig = make_figure(
        height=1,
        width=1.2,
        x_domains=x_domains,
        y_domains=y_domains,
    )

    ym = 0
    title_txt = ''
    for i, tid in enumerate([tid_WT, tid_CX]):

        data: BootstrapOutput = cluster_data[tid]
        x = data.bins
        y = data.firing_rate
        y_neg = data.firing_rate_ci_low
        y_pos = data.firing_rate_ci_high
        if np.max(y_pos) > ym:
            ym = np.max(y_pos)

        if i == 0:
            clr = 'rgba(0, 0, 0, 1)'
            clr_a = 'rgba(0, 0, 0, 0.5)'
        else:
            clr = 'rgba(0, 0, 255, 1)'
            clr_a = 'rgba(0, 0, 255, 0.5)'

        fig.add_scatter(x=x, y=y_neg, mode='lines', line=dict(width=0.001, color=clr_a), showlegend=False)
        fig.add_scatter(x=x, y=y_pos, mode='lines', line=dict(width=0.001, color=clr_a), showlegend=False,
                        fill='tonexty', fillcolor=clr_a)
        fig.add_scatter(x=x, y=y, mode='lines', line=dict(width=2, color=clr), showlegend=False)

        ex_bins = data.excitation_bins
        in_bins = data.inhibition_bins

        if ex_bins is not None:
            fig.add_scatter(x=x[ex_bins], y=y[ex_bins], mode='markers', marker=dict(color=clr, symbol='diamond', size=8), showlegend=False)
        if in_bins is not None:
            fig.add_scatter(x=x[in_bins], y=y[in_bins], mode='markers', marker=dict(color=clr, symbol='diamond', size=8), showlegend=False)
            title_txt += f'{["WT", "CX"][i]}: {data.baseline_firing_rate_mean:.4f} Hz '


    fig.update_xaxes(
        tickvals=np.arange(-200, 200, 100),
        title_text='time [ms]'
    )

    ym = ym + 0.1 * ym
    fig.update_yaxes(
        range=[-0.05*ym, ym]
    )

    utils.update_subplot_titles(fig, x_domains, y_domains, subplot_titles={(1,1): title_txt})
    save_fig(fig, savename, display=False)


def main():
    # Load dataset + dump as pickle to speedup future data loading
    data_io = DataIO(dataset_dir)

    all_frates = {
        'WT': {'0_WT': [], '1_CX': [], '2_both': []},
        'CX': {'0_WT': [], '1_CX': [], '2_both': []}}

    #
    #           ------------------- SECTION 1: SELECT CLUSTERS -----------------
    #
    for session_id, (rid_WT, rid_CX) in data_list.items():
        data_io.load_session(session_id, load_pickle=True, load_waveforms=False)
        # data_io.dump_as_pickle()

        loadname = dataset_dir / f'{data_io.session_id}_cells.csv'
        cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
        pref_ec = detect_preferred_electrode(data_io, cells_df)

        # Select EC during CNQX recording if there is one, otherwise during WT
        clusters_to_plot = pref_ec[rid_CX]['pa_dmd_timing_full_field'].dropna().copy()
        for i, r in pref_ec[rid_WT]['pa_dmd_timing_full_field'].iterrows():
            if i not in clusters_to_plot.index.values and pd.notna(r.ec):
                clusters_to_plot.at[i, 'ec'] = r.ec

        for cid, cinfo in clusters_to_plot.iterrows():
            for rtype, red_ic in zip(['WT', 'CX'], [rid_WT, rid_CX]):
                train_df = data_io.train_df.query(
                    f'rec_id ==  "{red_ic}" and '
                    f'laser_burst_duration == {LASER_BURST_DURATION} and '
                    f'laser_pulse_repetition_rate == {LASER_PRR} and '
                    f'laser_power == {LASER_POWER} and '
                    f'has_dmd == False and '
                    f'electrode == {cinfo.ec}'
                )
                assert len(train_df) == 1
                clusters_to_plot.at[cid, f'tid_{rtype}'] = train_df.iloc[0].name
                if cells_df.loc[cid][train_df.iloc[0].name, 'is_excited'] or cells_df.loc[cid][train_df.iloc[0].name, 'is_inhibited']:
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


        fig = make_figure(
            height=2,
            x_domains={1: [[0.1, 0.5], [0.55, 0.95]]},
            y_domains={1: [[0.1, 0.9], [0.1, 0.9]]},
            subplot_titles={1: [f'PA power: {LASER_POWER:.0f}, PA PRR: {LASER_PRR/1e3:.1f} kHz, PA duration: {20:.0f} ms', '']}
        )

        col = 1

        for rtype, red_ic in zip(['WT', 'CX'], [rid_WT, rid_CX]):
            frates, vlines = [], []
            last_lbl = '0_WT'

            for cid, cinfo in clusters_to_plot.iterrows():

                # if cells_df.loc[cid][tid]['laser_distance'] > MIN_D:
                #     continue

                cluster_data: Dict[str, BootstrapOutput] = load_obj(dataset_dir / 'bootstrapped' / f'bootstrap_{cid}.pkl')

                tid = cinfo[f'tid_{rtype}']
                mean_fr = cluster_data[tid].firing_rate
                bin_centres = cluster_data[tid].bins
                baseline_mask = (bin_centres >= baseline_t0) & (bin_centres < baseline_t1)

                frates.append(mean_fr)
                all_frates[rtype][cinfo.lbl].append(mean_fr)

                if last_lbl != cinfo.lbl:
                    vlines.append(len(frates))
                    last_lbl = cinfo.lbl

                # if rtype == 'WT':
                #     plot_single_cell_firing_rate(cluster_data, cinfo['tid_WT'], cinfo['tid_CX'],
                #                                  savename=figure_dir_analysis / 'heatmap_all_cells_WT_CNQX' / 'cells' / f'{cid}'
                #                                  )


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
                ),
                row=1,
                col=col,
            )

            fig.add_scatter(
                x=[0, 0, 20, 20],
                y=[-1, cell_fr.shape[0]+1, cell_fr.shape[0]+1, -1],
                mode='lines',
                line=dict(color='red', width=0.0001),
                fill='toself',
                fillcolor='rgba(179, 157, 219, 0.4)',
                showlegend=False,
                row=1,
                col=col,
            )

            lbls = clusters_to_plot.lbl.unique()
            yticks = [vlines[0] / 2, (vlines[1] - vlines[0]) / 2 + vlines[0], (cell_fr.shape[0] - vlines[1]) / 2 + vlines[1]]
            ylbls = ['WT-only', 'CNQX-only', 'both']
            for v in vlines:
                fig.add_scatter(
                    x=bin_centres,
                    y=np.ones_like(bin_centres) * (v-0.5),
                    mode='lines', line=dict(color='black', width=2),
                    showlegend=False,
                    row=1, col=col,
                )

            fig.update_xaxes(
                tickvals=np.arange(-200, 301, 50),
                title_text='time [ms]',
                row=1,
                col=col,
            )

            fig.update_yaxes(
                title_text = 'cell #' if col == 1 else '',
                range=[-0.5, cell_fr.shape[0]-0.5],
                tickvals=yticks if col == 1 else [],
                ticktext=ylbls if col == 1 else [],
                tickangle=-90,
                # tickvals=np.arange(0, len(labels)),
                # ticktext=labels,
                row=1,
                col=col,
            )
            col += 1

        savename = figure_dir_analysis / 'heatmap_all_cells_WT_CNQX' / f'{session_id}'
        save_fig(fig=fig, savename=savename, display=True)

    # Make a grand average figure
    fig = make_figure(
        height=2,
        x_domains={1: [[0.1, 0.5], [0.55, 0.95]]},
        y_domains={1: [[0.1, 0.9], [0.1, 0.9]]},
        subplot_titles={
            1: [f'PA power: {LASER_POWER:.0f}, PA PRR: {LASER_PRR / 1e3:.1f} kHz, PA duration: {20:.0f} ms', '']}
    )

    lbl ='0_WT'
    lbls = ['0_WT', '1_CX', '2_both']

    for col, dtype in enumerate(['WT', 'CX']):
        all_cell_fr = None
        vlines = []
        for lbl in lbls:
            cell_fr = np.array(all_frates[dtype][lbl])
            if all_cell_fr is None:
                all_cell_fr = cell_fr
            else:
                all_cell_fr = np.vstack((all_cell_fr, cell_fr))
            vlines.append(cell_fr.shape[0])


        all_cell_fr = all_cell_fr / (binwidth / 1000)

        baseline = all_cell_fr[:, baseline_mask]

        baseline_mean = baseline.mean(axis=1, keepdims=True)
        baseline_std = baseline.std(axis=1, keepdims=True)

        # baseline_std[baseline_std == 0] = 1  # avoid divide-by-zero
        idx = np.where(baseline != 0)[0]
        all_cell_fr[idx] = (all_cell_fr[idx] - baseline_mean[idx]) / baseline_std[idx]

        # 1. Clip the data to your strict boundaries
        clipped_fr = np.clip(all_cell_fr, zmin, zmax)

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

        fig.add_heatmap(
            z=nonlinear_fr,
            x=bin_centres,
            y=np.arange(nonlinear_fr.shape[0]),
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
            ),
            row=1,
            col=col+1,
        )

        fig.add_scatter(
            x=[0, 0, 20, 20],
            y=[-1, nonlinear_fr.shape[0] + 1, nonlinear_fr.shape[0] + 1, -1],
            mode='lines',
            line=dict(color='red', width=0.0001),
            fill='toself',
            fillcolor='rgba(179, 157, 219, 0.7)',
            showlegend=False,
            row=1,
            col=col+1,
        )

        yticks = [
            vlines[0] / 2,
            (vlines[1] / 2) + vlines[0],
            (vlines[2] / 2 + vlines[0] + vlines[1])
        ]
        vl_plot = [
            vlines[0],
            vlines[0] + vlines[1],
        ]
        ylbls = [f'WT-only (n={vlines[0]})', f'CNQX-only (n={vlines[1]})', f'both (n={vlines[2]})']
        for v in vl_plot:
            fig.add_scatter(
                x=bin_centres,
                y=np.ones_like(bin_centres) * (v - 0.5),
                mode='lines', line=dict(color='black', width=2),
                showlegend=False,
                row=1, col=col+1,
            )

        fig.update_xaxes(
            tickvals=np.arange(-200, 301, 50),
            title_text='time [ms]',
            row=1,
            col=col+1,
        )

        fig.update_yaxes(
            title_text='cell #' if col == 0 else '',
            range=[-0.5, nonlinear_fr.shape[0] - 0.5],
            tickvals=yticks if col == 0 else [],
            ticktext=ylbls if col == 0 else [],
            tickangle=-90,
            # tickvals=np.arange(0, len(labels)),
            # ticktext=labels,
            row=1,
            col=col+1,
        )

    savename = figure_dir_analysis / 'heatmap_all_cells_WT_CNQX' / f'all'
    save_fig(fig=fig, savename=savename, display=True)

if __name__ == '__main__':
    main()

