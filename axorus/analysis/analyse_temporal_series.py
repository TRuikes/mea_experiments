import pandas as pd
from pathlib import Path
from axorus.preprocessing.project_colors import ProjectColors
from axorus.data_io import DataIO
import utils
import numpy as np
import axorus.analysis.figure_library as fl
from scipy.stats import poisson


data_dir = Path(r'E:\Axorus\dataset_series_3')
figure_dir = Path(r'C:\Axorus\figures\temporal_series')

sessions = (
    '241108_A',
    '241211_A',
    '241213_A',
)

train_periods = [25, 50,  100,  250,  500, 1000]


def main():
    read_session_parameters()
    selected_cells_df = select_cells()
    compute_single_cell_success_rate(selected_cells_df)

    plot_single_cell_responses_per_trial(selected_cells_df)
    success_rate_per_session(selected_cells_df)


def success_rate_per_session(selected_cells_df):

    fig = utils.make_figure(
        width=0.6,
        height=0.6,
        x_domains={1: [[0.1, 0.9]]},
        y_domains={1: [[0.1, 0.9]]},
    )

    print(f'\n\nsuccess rate per session:')

    success_rate_mean = pd.DataFrame()
    success_rate_se = pd.DataFrame()

    for sid, sdf in selected_cells_df.groupby('sid'):

        txt = f'{sid} | '

        for tp in train_periods:
            val = sdf[tp].mean()
            se = sdf[tp].std() / np.sqrt(sdf.shape[0])
            txt += f'{tp:.0f}: {val:.2f}\t'

            success_rate_mean.at[sid, tp] = val
            success_rate_se.at[sid, tp] = se

        print(txt)

        xticks = success_rate_mean.loc[sid].keys()[::-1]
        x = np.arange(0, xticks.size)
        y = success_rate_mean.loc[sid].values[::-1]

        se = success_rate_se.loc[sid].values[::-1]
        fig.add_scatter(
            x=x, y=y,
            error_y=dict(array=se, width=1),
            mode='lines+markers',
            line=dict(width=1),
            marker=dict(size=4,),
            showlegend=False,
        )

    fig.update_xaxes(
        tickvals=x,
        ticktext=xticks,
        title_text='burst interval [ms]',
    )
    fig.update_yaxes(
        title_text='stimulation success rate',
        tickvals=np.arange(0, 1.2, 0.2),

    )
    savename = figure_dir / 'success_rate_per_session'
    utils.save_fig(fig, savename)


def plot_single_cell_responses_per_trial(selected_cells_df):
    n_cols = 3
    n_rows = 2
    x_domains = {}
    y_domains = {}
    x_offset = 0.05
    x_spacing = 0.01
    x_width = (1 - ((n_cols - 1) * x_spacing) - 2 * x_offset) / n_cols
    y_offset = 0.1
    y_spacing = 0.15
    y_height = (1 - ((n_rows - 1) * y_spacing) - 2 * y_offset) / n_rows
    clrs = ProjectColors()

    xmax = 200
    xmin = -25

    for row_i in range(n_rows):
        y1 = 1 - y_offset - row_i * (y_spacing + y_height)
        y_domains[row_i + 1] = [[y1 - y_height, y1] for _ in range(n_cols)]
        x_domains[row_i + 1] = []
        for col_i in range(n_cols):
            x0 = x_offset + col_i * (x_spacing + x_width)
            x_domains[row_i + 1].append([x0, x0 + x_width])

    pos = {
        1000: dict(row=1, col=1),
        500: dict(row=1, col=2),
        250: dict(row=1, col=3),
        100: dict(row=2, col=1),
        50: dict(row=2, col=2),
        25: dict(row=2, col=3),
    }

    # Load dataset
    data_io = DataIO(data_dir)

    # Find stats per session
    for sid, session_cells_df in selected_cells_df.groupby('sid'):
        data_io.load_session(sid, load_pickle=True, load_waveforms=False)

        # Find success rate for each included cell
        for cid, cinfo in session_cells_df.iterrows():

            # Select all bursts on the clusters' favourite electrode
            all_bursts = data_io.burst_df.query(f'protocol == "pa_temporal_series"'
                                                f'and electrode == {cinfo.ec}')

            cluster_data = utils.load_obj(data_dir / 'bootstrapped' / f'bootstrap_{cid}.pkl')

            fig = utils.make_figure(
                width=1,
                height=1.2,
                x_domains=x_domains,
                y_domains=y_domains,
                # equal_width_height='x',
                subplot_titles={
                    1: [1000, 500, 250],
                    2: [100, 50, 25],
                }
            )

            for tid, tdf in all_bursts.groupby('train_id'):
                train_period = tdf.iloc[0].train_period
                clr = clrs.train_period(train_period)
                p = pos[train_period]

                stimes = cluster_data[tid]['spike_times']

                x_plot, y_plot = [], []
                xp_sig, yp_sig = [], []

                for burst_i, sp in enumerate(stimes):
                    sp_to_plot = sp[(sp > 0) & (sp < train_period)]
                    x_plot.append(np.vstack([sp_to_plot, sp_to_plot, np.full(sp_to_plot.size, np.nan)]).T.flatten())
                    y_plot.append(np.vstack([np.ones(sp_to_plot.size) * burst_i,
                                             np.ones(sp_to_plot.size) * burst_i + 1, np.full(
                            sp_to_plot.size, np.nan)]).T.flatten())
                    is_sig = detect_burst_significance(sp, train_period)

                    if is_sig:
                        xp_sig.append([xmin, xmin, xmax, xmax, None])
                        yp_sig.append([burst_i, burst_i + 1, burst_i + 1, burst_i, None])

                x_plot = np.hstack(x_plot)
                y_plot = np.hstack(y_plot)

                fig.add_scatter(
                    x=x_plot, y=y_plot,
                    mode='lines', line=dict(color=clr, width=0.5),
                    showlegend=False,
                    **p,
                )
                fig.update_xaxes(
                    title_text='time [ms]' if p['row'] == 2 else '',
                    tickvals=np.arange(-50, xmax + 50, 50),
                    range=[xmin, xmax],
                    **p
                )

                fig.update_yaxes(
                    title_text='burst nr' if p['col'] == 1 else '',
                    range=[0, len(stimes)],
                    **p,
                )

            for tp in train_periods:
                if cinfo.loc[tp] > 0.5:
                    subdir = f'{tp}'
                    break

            savename = figure_dir / 'single_cell_responses_per_trial' / subdir / cid
            utils.save_fig(fig, savename, display=False)


def compute_single_cell_success_rate(selected_cells_df):

    # Load dataset
    data_io = DataIO(data_dir)

    # Find stats per session
    for sid, session_cells_df in selected_cells_df.groupby('sid'):
        data_io.load_session(sid, load_pickle=True, load_waveforms=False)

        # Find success rate for each included cell
        for cid, cinfo in session_cells_df.iterrows():

            # Load pre extracted data (by analyse response script)
            cluster_data = utils.load_obj(data_dir / 'bootstrapped' / f'bootstrap_{cid}.pkl')

            # Select all bursts on the clusters' favourite electrode
            all_bursts = data_io.burst_df.query(f'protocol == "pa_temporal_series"'
                                            f'and electrode == {cinfo.ec}')

            # Detect per trial (e.g. different train periods) the success rate
            for tid, tinfo in all_bursts.groupby('train_id'):

                # Detect this trial burst period
                train_period = tinfo.train_period.values[0]

                # Get the spiketimes relative to stimulation times
                stimes = cluster_data[tid]['spike_times']
                n_bursts = len(stimes)

                # Count the nr of significant bursts
                n_sig = 0

                # detect per burst if its significant
                for burst_i, sp in enumerate(stimes):
                    sp_to_plot = sp[(sp > 0) & (sp < train_period)]
                    is_sig = detect_burst_significance(sp_to_plot, train_period)
                    if is_sig:  n_sig += 1

                selected_cells_df.at[cid, train_period] = n_sig / n_bursts



        #     clr = clrs.train_period(train_period)
        #     p = pos[train_period]
        #
        #     stimes = cluster_data[tid]['spike_times']
        #
        #     x_plot, y_plot = [], []
        #     xp_sig, yp_sig = [], []
        #     n_sig = 0
        #
        #     for burst_i, sp in enumerate(stimes):
        #         sp_to_plot = sp[(sp > 0) & (sp < train_period)]
        #         x_plot.append(np.vstack([sp_to_plot, sp_to_plot, np.full(sp_to_plot.size, np.nan)]).T.flatten())
        #         y_plot.append(np.vstack([np.ones(sp_to_plot.size) * burst_i,
        #                                  np.ones(sp_to_plot.size) * burst_i + 1, np.full(
        #                 sp_to_plot.size, np.nan)]).T.flatten())
        #         is_sig = detect_burst_significance(sp, train_period)
        #
        #         if is_sig:
        #             xp_sig.append([xmin, xmin, xmax, xmax, None])
        #             yp_sig.append([burst_i, burst_i + 1, burst_i + 1, burst_i, None])
        #             n_sig += 1
        #
        #     x_plot = np.hstack(x_plot)
        #     y_plot = np.hstack(y_plot)
        #
        #     n_bursts = len(stimes)
        #
        #     if train_period == 1000:
        #         perc_success = n_sig / n_bursts

    return selected_cells_df


def select_cells():
    print(f'\n\n\nSelecting cells')
    print(f'Inclusion criteria:')
    print(f'\t-significant response at train period = 1000')
    print(f'\t-if significant at multiple electrodes, '
          f'select the electrode with strongest response')

    data_io = DataIO(data_dir)

    # Placeholder for selected cells
    selected_cells_df = pd.DataFrame()

    for sid in sessions:
        data_io.load_session(sid, load_pickle=True, load_waveforms=False)
        loadname = data_dir / f'{sid}_cells.csv'
        data_io.load_session(sid, load_pickle=True)
        cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)

        trials = data_io.burst_df.query(f'protocol == "pa_temporal_series"')

        tp1000_tids = trials.query('train_period == 1000').train_id.unique()

        for cid, cinfo in cells_df.iterrows():
            df = pd.DataFrame()

            for tid in tp1000_tids:
                is_sig = cinfo[(tid, 'is_significant')]
                df.at[tid, 'is_sig'] = is_sig if not pd.isna(is_sig) else False
                df.at[tid, 'fr'] = cinfo[(tid, 'response_firing_rate')]

            df = df.loc[df.is_sig]
            if df.shape[0] == 0:
                continue

            max_stim = df.loc[df['fr'].idxmax()]
            selected_cells_df.at[cid, 'sid'] = sid
            selected_cells_df.at[cid, 'tid'] = max_stim.name
            selected_cells_df.at[cid, 'ec'] = trials.query(f'train_id == "{max_stim.name}"').electrode.values[0]

    return selected_cells_df


def read_session_parameters():
    data_io = DataIO(data_dir)

    for sid in sessions:
        data_io.load_session(sid, load_pickle=True, load_waveforms=False)
        loadname = data_dir / f'{sid}_cells.csv'
        data_io.load_session(sid, load_pickle=True)

        trials = data_io.burst_df.query(f'protocol == "pa_temporal_series"')
        n_trials = trials.shape[0]

        burst_durations = trials.burst_duration.unique()
        assert len(burst_durations) == 1
        pulse_rep_fs = trials.repetition_frequency.unique()
        assert len(pulse_rep_fs) == 1, pulse_rep_fs
        pulse_energies = trials.e_pulse.unique()
        assert len(pulse_energies) == 1, pulse_energies
        train_periods = np.sort(trials.train_period.unique())

        print(f'\n\n{sid}')

        print(f'stimulation parameters:')
        print(f'\tn trails: {n_trials}')
        print(f'\tburst duration: {burst_durations[0]:.0f} ms')
        print(f'\tpulse repetition frequency: {pulse_rep_fs[0]:.0f} Hz')
        print(f'\tpulse energy: {pulse_energies[0]:.0f} uJ')
        print(f'stimulated at train periods: {train_periods}')


def poisson_test(n_baseline, t_baseline, n_stim, t_stim):
    # Baseline rate
    lambda_baseline = n_baseline / t_baseline

    # Compute p-value (one-tailed test)
    p_value = 1 - poisson.cdf(n_stim - 1, lambda_baseline * t_stim)
    return p_value


def detect_burst_significance(spiketrain, t_1):
    b_0 = -50
    b_1 = 0

    t_0 = 0
    # t_1 = 150
    bin_size = b_1 - b_0
    bin_hw = bin_size / 2
    t_step = 10

    bns = np.arange(t_0+bin_hw, t_1 - bin_hw, t_step)

    n_base = np.where((spiketrain >= b_0) & (spiketrain <= b_1))[0].size

    HAS_SIG_BIN = False
    for bin_i, bin_centre in enumerate(bns):
        t0 = bin_centre - bin_hw
        t1 = bin_centre + bin_hw
        n_stim = np.where((spiketrain >= t0) & (spiketrain < t1))[0].size

        p = poisson_test(n_base, bin_size / 1000, n_stim, bin_size / 1000)

        if p < 0.05:
            HAS_SIG_BIN = True
            break

    return HAS_SIG_BIN


if __name__ == '__main__':
    main()