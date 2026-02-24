"""
For analysis of individual sessions see:
plot_250520_A and plot_250527_A in /analysis

this panel goes to figure 2 in the axorus 2025 publication

"""
import sys
sys.path.append('.')
from pathlib import Path
from axorus.data_io import DataIO
import numpy as np
from utils import make_figure, save_fig
from scipy.stats import wilcoxon, ttest_ind, friedmanchisquare
from axorus.preprocessing.project_colors import ProjectColors
import utils
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import io


data_dir = Path(r'C:\axorus\dataset')
# data_dir = Path(r'E:\Axorus\dataset_series_3')
figure_dir = Path(r'C:\axorus\figures\paper\Figure_3')
# figure_dir = Path(r'E:\Axorus\Figures') / 'lap4analysis'
session_ids = ('250520_A', '250527_A')  # Dataset sessions to include

inclusion_range = 200  # diameter of cells to include from laser source


def gather_figure_stats(update=False):
    """
    Gather the statistics for visualization
    :return: df: pandas dataframe containing response statistics
    for each individual cell
    """
    savename = figure_dir / 'fig2j_data.csv'
    if savename.is_file() and not update:
        df_out = pd.read_csv(savename,
                         header=0, index_col=0)
        print(f'Loaded presaved data')
        return df_out

    df_out = pd.DataFrame()  # Placeholder for output

    data_io = DataIO(data_dir)

    # Load data per session
    print(f'Loading data for {len(session_ids)} sessions...')
    for sid in session_ids:
        print(f'\t{sid}')

        # Load dataset
        data_io.load_session(sid, load_pickle=True)

        # Load analysis results
        loadname = data_dir / f'{sid}_cells.csv'
        cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)

        # Set stimulation sites and blockers for this session
        if sid == '250520_A':
            stimulation_sites = [[164], [155], [63], [217, 100]]
            blockers = ['noblocker', 'lap4', 'lap4acet', 'washout']  # this session also has lap4acet

        elif sid == '250527_A':
            stimulation_sites = [[86], [93], [129], [207], [220]]
            blockers = ['noblocker', 'lap4', 'lap4acet', 'washout']

        else:
            raise ValueError('did not include this session?')

        # Detect the preferred stimulation site for each cell
        # The preferred stim site is defined as the stim electrode
        # with the most significant trials
        pref_ec_dict = {}
        for cluster_id in data_io.cluster_df.index.values:

            pref_ec = None
            n_sig_pref_ec = None

            for stimulation_site in stimulation_sites:
                df = data_io.burst_df.query(f'electrode in {stimulation_site} and blockers == "noblocker"')
                tids = df.train_id.unique()
                n_sig = 0
                for tid in tids:
                    if cells_df.loc[cluster_id, (tid, 'is_significant')] == True:
                        n_sig += 1

                if n_sig > 1:
                    if pref_ec is None or n_sig > n_sig_pref_ec:
                        pref_ec = stimulation_site
                        n_sig_pref_ec = n_sig

            pref_ec_dict[cluster_id] = pref_ec

        # Load response data per cluster
        for cluster_id, electrode in pref_ec_dict.items():
            if electrode is None:
                continue

            # Load the response data
            for blocker in blockers:
                d_select = data_io.burst_df.query('electrode in @electrode and '
                                                  'blockers == @blocker').copy()
                tid = d_select.loc[d_select['duty_cycle'].idxmax()].train_id

                df_out.at[cluster_id, f'{blocker} baseline'] = cells_df.loc[cluster_id, tid].baseline_firing_rate
                df_out.at[cluster_id, f'{blocker} response'] = cells_df.loc[cluster_id, tid].response_firing_rate
                df_out.at[cluster_id, f'{blocker} response_latency'] = cells_df.loc[cluster_id, tid].response_latency

                laser_x = d_select.query('train_id == @tid').iloc[0].laser_x
                laser_y = d_select.query('train_id == @tid').iloc[0].laser_y

                cluster_x = data_io.cluster_df.loc[cluster_id, 'cluster_x']
                cluster_y = data_io.cluster_df.loc[cluster_id, 'cluster_y']
                d = np.sqrt((laser_x - cluster_x) ** 2 + (laser_y - cluster_y) ** 2)

                df_out.at[cluster_id, f'laser_distance'] = d

                tids = d_select.train_id.unique()
                is_sig = False
                for tid in tids:
                    if cells_df.loc[cluster_id, (tid, 'is_significant')] is True:
                        is_sig = True
                df_out.at[cluster_id, f'{blocker} is_sig'] = is_sig
    print(f'finished (saved data in {savename})!\n')

    df_out.to_csv(savename)
    return df_out


def gather_stats_n_cells_per_stimsite(update):
    """
    Gather the statistics for visualization
    :return: df: pandas dataframe containing response statistics
    for each individual cell
    """
    savename = figure_dir / 'fig3_n_cells_per_stimsite.pkl'

    if savename.exists() and not update:
        data_out = utils.load_obj(savename)
        return data_out

    data_io = DataIO(data_dir)

    # Load data per session
    print(f'Loading data for {len(session_ids)} sessions...')
    data_out = {}

    for sid in session_ids:
        print(f'\t{sid}')

        # Load dataset
        data_io.load_session(sid, load_pickle=True)

        # Load analysis results
        loadname = data_dir / f'{sid}_cells.csv'
        cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)

        # Set stimulation sites and blockers for this session
        if sid == '250520_A':
            stimulation_sites = [[164], [155], [63], [217, 100]]
            blockers = ['noblocker', 'lap4', 'lap4acet', 'washout']  # this session also has lap4acet

        elif sid == '250527_A':
            stimulation_sites = [[86], [93], [129], [207], [220]]
            blockers = ['noblocker', 'lap4', 'lap4acet', 'washout']

        else:
            raise ValueError('did not include this session?')

        # Detect the preferred stimulation site for each cell
        # The preferred stim site is defined as the stim electrode
        # with the most significant trials

        for stim_i, stim_site in enumerate(stimulation_sites):
            df_out = pd.DataFrame()

            for blocker in blockers:

                df = data_io.burst_df.query(f'electrode in {stim_site} and blockers == "{blocker}"')
                tid = df.loc[df['duty_cycle'].idxmax()].train_id

                laser_x = df.query('train_id == @tid').iloc[0].laser_x
                laser_y = df.query('train_id == @tid').iloc[0].laser_y

                for cluster_id in data_io.cluster_df.index.values:

                    cluster_x = data_io.cluster_df.loc[cluster_id, 'cluster_x']
                    cluster_y = data_io.cluster_df.loc[cluster_id, 'cluster_y']
                    d = np.sqrt((laser_x - cluster_x) ** 2 + (laser_y - cluster_y) ** 2)
                    df_out.at[cluster_id, f'd'] = d

                    if cells_df.loc[cluster_id, (tid, 'is_significant')] is True:
                        df_out.at[cluster_id, f'{blocker}'] = True
                    else:
                        df_out.at[cluster_id, f'{blocker}'] = False

            data_out[f'{sid}_{stim_i}'] = df_out

    utils.save_obj(data_out, savename)
    return data_out



def run_friedman_wilcoxon(df, savename):
    output = io.StringIO()

    conditions = ['noblocker', 'lap4', 'lap4acet', 'washout']
    latency_conditions = ['noblocker', 'lap4', 'washout']

    # Columns for baseline and response (8 columns)
    baseline_response_cols = []
    for cond in conditions:
        baseline_response_cols.append(f'{cond} baseline')
        baseline_response_cols.append(f'{cond} response')
    df_clean_baseline_response = df.dropna(subset=baseline_response_cols)

    # Columns for response_latency, only 3 conditions (exclude lap4acet)
    latency_cols = [f'{cond} response_latency' for cond in latency_conditions]
    df_clean_latency = df.dropna(subset=latency_cols)

    output.write(f'Data from {df_clean_baseline_response.shape[0]} cells\n\n')

    # Print Firing Rate mean and std values for all groups
    output.write("Firing Rate Mean and STD for all groups and condition\n")
    for c in conditions:
        for e in ['baseline', 'response']:
            data = df_clean_baseline_response[f'{c} {e}']
            mean = np.mean(data)
            se = np.std(data) / np.sqrt(len(data))
            output.write(f"{c} {e}: {mean:.2f}, {se:.2f} Hz (mean + SE)\n")
        output.write(f'\n')

    # Print Latency mean and SE values for all groups
    output.write("Latency Mean and STD for all groups and condition\n")
    for c in conditions:
        data = df[f'{c} response_latency']
        data = data[pd.notna(data)]
        mean = np.mean(data)
        se = np.std(data) / np.sqrt(len(data))
        output.write(f"{c} latency: {mean:.2f}, {se:.2f} ms (mean + SE)\n")

        n_45min = np.where(data < 45)[0].size
        n_45plus = np.where(data >= 45)[0].size
        output.write(f'{c} % cells faster than 45 ms: {100*n_45min/(n_45min + n_45plus):.0f} %\n')


    # Friedman test on all baseline + response data
    data_baseline_response = [df_clean_baseline_response[col] for col in baseline_response_cols]
    stat1, p1 = friedmanchisquare(*data_baseline_response)
    output.write(f"Friedman test on all baseline and response data:\nStatistic = {stat1:.2f}, p = {p1:.2e}\n\n")

    # Friedman test on response_latency (noblocker, lap4, washout)
    data_latency = [df_clean_latency[col] for col in latency_cols]
    stat2, p2 = friedmanchisquare(*data_latency)
    output.write(
        f"Friedman test on response_latency (noblocker, lap4, washout):\nStatistic = {stat2:.2f}, p = {p2:.2e}\n\n")

    # Wilcoxon baseline vs response within each condition
    output.write("Wilcoxon tests (baseline vs response within each condition):\n")
    pvals_baseline = []
    effect_sizes_baseline = []
    stat_values_baseline = []
    labels_baseline = []
    for cond in conditions:
        stat_w, pval = wilcoxon(df_clean_baseline_response[f'{cond} baseline'],
                                df_clean_baseline_response[f'{cond} response'])
        n = len(df_clean_baseline_response[f'{cond} baseline'])
        mean_w = n * (n + 1) / 4
        std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        Z = (stat_w - mean_w) / std_w
        r = abs(Z) / np.sqrt(n)
        pvals_baseline.append(pval)
        effect_sizes_baseline.append(r)
        stat_values_baseline.append(stat_w)
        labels_baseline.append(cond)
    corrected_baseline = multipletests(pvals_baseline, method='holm')[1]
    for cond, pval_corr, r, stat_w in zip(labels_baseline, corrected_baseline, effect_sizes_baseline,
                                          stat_values_baseline):
        star = ' *' if pval_corr < 0.01 else ''
        output.write(
            f"{cond}: Statistic = {stat_w:.2f}, corrected p = {pval_corr:.2e}, effect size r = {r:.2f}{star}\n")

    # Wilcoxon response vs response between conditions
    output.write("\nWilcoxon tests (response vs response between conditions):\n")
    pvals_response = []
    effect_sizes_response = []
    stat_values_response = []
    labels_response = []
    pairs = list(combinations(conditions, 2))
    for c1, c2 in pairs:
        stat_w, pval = wilcoxon(df_clean_baseline_response[f'{c1} response'],
                                df_clean_baseline_response[f'{c2} response'])
        n = len(df_clean_baseline_response[f'{c1} response'])
        mean_w = n * (n + 1) / 4
        std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        Z = (stat_w - mean_w) / std_w
        r = abs(Z) / np.sqrt(n)
        pvals_response.append(pval)
        effect_sizes_response.append(r)
        stat_values_response.append(stat_w)
        labels_response.append((c1, c2))
    corrected_response = multipletests(pvals_response, method='holm')[1]
    for (c1, c2), pval_corr, r, stat_w in zip(labels_response, corrected_response, effect_sizes_response,
                                              stat_values_response):
        star = ' *' if pval_corr < 0.01 else ''
        output.write(
            f"{c1} vs {c2}: Statistic = {stat_w:.2f}, corrected p = {pval_corr:.2e}, effect size r = {r:.2f}{star}\n")

    # Wilcoxon response_latency comparisons (only between noblocker, lap4, washout)
    output.write("\nWilcoxon tests (response_latency between noblocker, lap4, and washout):\n")
    pvals_latency = []
    effect_sizes_latency = []
    stat_values_latency = []
    labels_latency = []
    pairs_latency = list(combinations(latency_conditions, 2))
    for c1, c2 in pairs_latency:
        idx = np.where(pd.notna(df[f'{c1} response_latency']) &
                       pd.notna(df[f'{c2} response_latency']))[0]
        stat_w, pval = wilcoxon(df[f'{c1} response_latency'].values[idx],
                                df[f'{c2} response_latency'].values[idx])

        n = len(df[f'{c1} response_latency'].values[idx])
        mean_w = n * (n + 1) / 4
        std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        Z = (stat_w - mean_w) / std_w
        r = abs(Z) / np.sqrt(n)
        pvals_latency.append(pval)
        effect_sizes_latency.append(r)
        stat_values_latency.append(stat_w)
        labels_latency.append((c1, c2))
    corrected_latency = multipletests(pvals_latency, method='holm')[1]
    for (c1, c2), pval_corr, r, stat_w in zip(labels_latency, corrected_latency, effect_sizes_latency,
                                              stat_values_latency):
        star = ' *' if pval_corr < 0.01 else ''
        output.write(
            f"{c1} vs {c2}: Statistic = {stat_w:.2f}, corrected p = {pval_corr:.2e}, effect size r = {r:.2f}{star}\n")

    result_text = output.getvalue()
    print(result_text)
    with open(savename, 'w') as f:
        f.write(result_text)


def plot_boxplots(df):

    # ## FIGURE SETUP
    n_rows = 1
    n_cols = 1

    y_top = 0.1
    y_bottom = 0.1
    yspacing = 0.1  # y space between rows

    xoffset = 0.1  # x space left and right of plots
    xspacing = 0.1  # x space between columns

    yheight = (1 - y_bottom - y_top - yspacing * (n_rows - 1))  # height of each plot
    rel_heights = [1, 1, 1]

    # Generate x and y spacing for all the subplots
    y_domains = dict()
    y1 = 1 - y_top
    for i in range(n_rows):
        row_h = yheight * rel_heights[i]
        y0 = y1 - row_h
        y_domains[i + 1] = [[y0, y1] for j in range(n_cols)]

        y1 -= (row_h + yspacing)

    xwidth = (1 - (n_cols - 1) * xoffset - xspacing - 0.05) / n_cols
    sx = [[xoffset + (xspacing + xwidth) * i, xoffset + (xspacing + xwidth) * i + xwidth] for i in range(n_cols)]
    clrs = ProjectColors()

    # Generate the figure
    fig = make_figure(
        width=0.3, height=0.6,
        x_domains={
            1: sx,
        },
        y_domains=y_domains,
        subplot_titles={
            1: ['', '', ],
        },
    )

    xpos = [0, 1, 3, 4, 6, 7, 9, 10]
    xdata = ['noblocker baseline', 'noblocker response',
             'lap4 baseline', 'lap4 response',
             'lap4acet baseline', 'lap4acet response',
             'washout baseline', 'washout response']

    n_pts = df.shape[0]

    box_specs = dict(
        name='LE',
        boxpoints='all',
        marker=dict(color=clrs.animal_color('LE', 1, 1), size=2),
        line=dict(color=clrs.animal_color('LE', 1, 1), width=1.5),
        showlegend=False,
    )

    for xp, xd in zip(xpos, xdata):
        fig.add_box(
            x=np.ones(n_pts) * xp,
            y=df[xd].values,
            **box_specs,
        )

    fig.update_yaxes(
        range=[0, 280],
        title_text=f'fr. [Hz]',
        tickvals=np.arange(0, 300, 50),
    )
    fig.update_xaxes(
        tickvals=[0.5, 3.5, 6.5],
        ticktext=['noblocker', 'lap4', 'lap4acet', 'washout']
,
    )

    savename = figure_dir / 'Figure_3_lap4_response_strength'
    save_fig(fig, savename, formats=['png', 'svg'], scale=3)


def plot_latencies(df):

    # ## FIGURE SETUP
    n_rows = 1
    n_cols = 1

    y_top = 0.1
    y_bottom = 0.1
    yspacing = 0.1  # y space between rows

    xoffset = 0.1  # x space left and right of plots
    xspacing = 0.1  # x space between columns

    yheight = (1 - y_bottom - y_top - yspacing * (n_rows - 1))  # height of each plot
    rel_heights = [1, 1, 1]

    # Generate x and y spacing for all the subplots
    y_domains = dict()
    y1 = 1 - y_top
    for i in range(n_rows):
        row_h = yheight * rel_heights[i]
        y0 = y1 - row_h
        y_domains[i + 1] = [[y0, y1] for j in range(n_cols)]

        y1 -= (row_h + yspacing)

    xwidth = (1 - (n_cols - 1) * xoffset - xspacing - 0.05) / n_cols
    sx = [[xoffset + (xspacing + xwidth) * i, xoffset + (xspacing + xwidth) * i + xwidth] for i in range(n_cols)]
    clrs = ProjectColors()

    # Generate the figure
    fig = make_figure(
        width=0.3, height=0.6,
        x_domains={
            1: sx,
        },
        y_domains=y_domains,
        subplot_titles={
            1: ['', '', ],
        },
    )

    xpos = [0, 2, 4]
    xdata = ['noblocker response_latency',
             'lap4 response_latency',
             # 'lap4acet response_latency',
             'washout response_latency']

    n_pts = df.shape[0]

    box_specs = dict(
        name='LE',
        boxpoints='all',
        marker=dict(color=clrs.animal_color('LE', 1, 1), size=2),
        line=dict(color=clrs.animal_color('LE', 1, 1), width=1.5),
        showlegend=False,
    )

    for xp, xd in zip(xpos, xdata):
        fig.add_box(
            x=np.ones(n_pts) * xp,
            y=df[xd].values,
            **box_specs,
        )

    fig.add_scatter(
        x=[-0.5, 4.5], y=[45, 45],
        mode='lines', line=dict(color='black', dash='2px', width=1),
        showlegend=False,
    )

    fig.update_yaxes(
        range=[0, 200],
        title_text=f'latency [ms]',
        tickvals=np.arange(0, 200, 50),
    )
    fig.update_xaxes(
        tickvals=xpos,
        ticktext=['no blocker', 'lap4', 'lap4acet', 'washout'],
    )

    savename = figure_dir / 'lap4acet_latencies'
    save_fig(fig, savename, formats=['png', 'svg'], scale=3)

    savename = figure_dir / 'latency_data.csv'
    df.to_csv(savename)
    print(f'saved latency data: {savename}')


def plot_frac_cells_responding(df):

    clrs = ProjectColors()

    # Figure setup
    n_rows = 1
    n_cols = 1

    y_top = 0.1
    y_bottom = 0.1
    yspacing = 0.1  # y space between rows

    xoffset = 0.1  # x space left and right of plots
    xspacing = 0.05  # x space between columns

    yheight = (1 - y_bottom - y_top - yspacing * (n_rows - 1))  # height of each plot
    rel_heights = [1]

    # Generate x and y spacing for all the subplots
    y_domains = dict()
    y1 = 1 - y_top
    for i in range(n_rows):
        row_h = yheight * rel_heights[i]
        y0 = y1 - row_h
        y_domains[i + 1] = [[y0, y1] for j in range(n_cols)]

        y1 -= (row_h + yspacing)

    xwidth = (1 - (n_cols - 1) * xoffset - xspacing - 0.05) / n_cols
    sx = [[xoffset + (xspacing + xwidth) * i, xoffset + (xspacing + xwidth) * i + xwidth] for i in range(n_cols)]


    data_plot = pd.DataFrame()

    blockers = ['noblocker', 'lap4', 'lap4acet', 'washout']

    for blocker in blockers:

        n_cells_tot = 0
        n_cells_resp = 0

        for k, d in df.items():
            d2 = d.query(f'd < {inclusion_range}')
            n_cells_tot += d2.shape[0]
            n_cells_resp += d2[blocker].sum()

        data_plot.at[blocker, 'n_cells'] = n_cells_tot
        data_plot.at[blocker, 'n_cells_resp'] = n_cells_resp
        data_plot.at[blocker, 'perc'] = (n_cells_resp / n_cells_tot) * 100

    txt = ''
    for i, r in data_plot.iterrows():
        txt += f'{i}:({2} retinas, {len(df.keys()):.0f} stimsites), '
        txt += f'{r.n_cells} cells in stim range: '
        txt += f'{r.n_cells_resp:.0f}, (={r.perc:.0f})%\n'

    savename = figure_dir / 'frac_responding_lap4.txt'

    with open(savename, 'w') as f:
        f.write(txt)

    fig_percentage = make_figure(
        width=0.25, height=0.6,
        x_domains={
            1: sx,
        },
        y_domains=y_domains,
        subplot_titles={
            1: ['', '',],
        },
    )

    xpos = [0, 2, 4, 6]

    for xp, blocker in zip(xpos, blockers):

        y = []
        for k, d in df.items():
            d2 = d.query(f'd < {inclusion_range}')
            y.append(d2[blocker].sum() / d2.shape[0])

        fig_percentage.add_box(
            x=np.ones(len(y)) * xp,
            # x=np.ones_like(n_cells_per_stimsite) * 0 if animal == 'LE' else 1,
            y=[f * 100 for f in y],
            boxpoints='all',
            showlegend=False,
            marker=dict(color=clrs.animal_color('LE', 1, 1), size=3),
            line=dict(color=clrs.animal_color('LE', 1, 1), width=2),
        )

    fig_percentage.update_yaxes(
        range=[0, 100],
        title_text=f'% Cells modulated',
        tickvals=np.arange(0, 100, 20),
    )
    fig_percentage.update_xaxes(
        tickvals=xpos,
        ticktext=['no blocker', 'lap4', 'lap4acet', 'washout'],
    )

    savename = figure_dir / 'Figure_3_lap4_frac_responding'
    save_fig(fig_percentage, savename, formats=['png', 'svg'], scale=3)


def main():
    df = gather_figure_stats(update=False)
    df = df.query(f'laser_distance < {inclusion_range}')

    df_dist = gather_stats_n_cells_per_stimsite(update=False)

    statsfile = figure_dir / f'response_fr_stats_lap4.txt'

    run_friedman_wilcoxon(df, statsfile)

    plot_boxplots(df)

    plot_latencies(df)

    plot_frac_cells_responding(df_dist)


if __name__ == '__main__':
    main()