"""
For analysis of individual sessions see:
plot_250520_A and plot_250527_A in /analysis

this panel goes to figure 2 in the axorus 2025 publication

"""

import pandas as pd
from pathlib import Path

from axorus.data_io import DataIO
import numpy as np
from utils import make_figure, save_fig
from scipy.stats import wilcoxon
from axorus.preprocessing.project_colors import ProjectColors

figure_dir = Path(r'C:\Axorus\figures') / 'lap4analysis'
data_dir = Path(r'C:\axorus\tmp')  # Path to dataset
session_ids = ('250520_A', '250527_A')  # Dataset sessions to include

def gather_figure_stats(update=False):
    """
    Gather the statistics for visualization
    :return: df: pandas dataframe containing response statistics
    for each individual cell
    """
    savename = figure_dir / 'fig2j_data.csv'
    if savename.is_file() and not update:
        df = pd.read_csv(savename,
                         header=0, index_col=0)
        print(f'Loaded presaved data')
        return df

    df = pd.DataFrame()  # Placeholder for output

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
            blockers = ['noblocker', 'lap4', 'washout']  # this session also has lap4acet

        elif sid == '250527_A':
            stimulation_sites = [[86], [93], [129], [207], [220]]
            blockers = ['noblocker', 'lap4', 'washout']

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
                    if cells_df.loc[cluster_id, (tid, 'is_significant')] is True:
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

                df.at[cluster_id, f'{blocker} baseline'] = cells_df.loc[cluster_id, tid].baseline_firing_rate
                df.at[cluster_id, f'{blocker} response'] = cells_df.loc[cluster_id, tid].response_firing_rate
                df.at[cluster_id, f'{blocker} response_latency'] = cells_df.loc[cluster_id, tid].response_latency

                tids = d_select.train_id.unique()
                is_sig = False
                for tid in tids:
                    if cells_df.loc[cluster_id, (tid, 'is_significant')] is True:
                        is_sig = True
                df.at[cluster_id, f'{blocker} is_sig'] = is_sig
    print(f'finished (saved data in {savename})!\n')

    df.to_csv(savename)
    return df


def print_wilcoxon(d0, d1, tag1, tag2):

    idx = np.where(pd.notna(d0) & pd.notna(d1))[0]

    r, p = wilcoxon(d0[idx], d1[idx])
    d0_m = np.mean(d0[idx])
    d0_s = np.std(d0[idx])
    d1_m = np.mean(d1[idx])
    d1_s = np.std(d1[idx])
    print(f'{tag1} vs {tag2}:({d0_m:.0f} ({d0_s:.0f}), {d1_m:.0f} ({d1_s:.0f})) T = {r:.0f} (p={p:.3f})')


def print_stats(df):
    blockers = ['noblocker', 'lap4', 'washout']  # this session also has lap4acet

    print(f'\n\nLAP4 experiments')
    print(f'{df.shape[0]} cells, {2} retinas')

    n_sig_noblocker = df['noblocker is_sig'].sum()
    n_sig_lap4 = df['lap4 is_sig'].sum()
    n_sig_washout = df['washout is_sig'].sum()
    print(f'N sig: noblocker: {n_sig_noblocker:.0f}, lap4: {n_sig_lap4:.0f}, washout: {n_sig_washout:.0f}\n')
    print_wilcoxon(df['noblocker baseline'], df['noblocker response'], 'none baseline', 'none stim')
    print_wilcoxon(df['lap4 baseline'], df['lap4 response'], 'lap4 baseline', 'stim lap4')
    print_wilcoxon(df['washout baseline'], df['washout response'], 'washout baseline', 'washout stim')

    print('\n')

    print_wilcoxon(df['noblocker response'], df['lap4 response'], 'noblocker response', 'lap4 response')
    print_wilcoxon(df['lap4 response'], df['washout response'], 'lap4 response', 'washout response')
    print_wilcoxon(df['noblocker response'], df['washout response'], 'noblocker response', 'washout response')


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
             'washout baseline', 'washout response']

    xlbl = ['no blocker', 'lap4', 'washout']

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
        ticktext=['no blocker', 'lap4', 'washout'],
    )

    savename = figure_dir / 'paper' / f'Figure_2' / 'Figure_2J_lap4acet'
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

    xpos = [0, 1, 3, 4, 6, 7, 9, 10]
    xdata = ['noblocker baseline', 'noblocker response',
             'lap4 baseline', 'lap4 response',
             'washout baseline', 'washout response']

    xlbl = ['no blocker', 'lap4', 'washout']

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
        ticktext=['no blocker', 'lap4', 'washout'],
    )

    savename = figure_dir / 'paper' / f'Figure_2' / 'Figure_2J_lap4acet'
    save_fig(fig, savename, formats=['png', 'svg'], scale=3)


def main():
    df = gather_figure_stats(update=False)

    print_stats(df)
    # plot_boxplots(df)
    plot_latencies(df)

if __name__ == '__main__':
    main()