from sonogenetics.analysis.analysis_tools import detect_preferred_electrode, params_per_protocol, params_abbreviation
from sonogenetics.analysis.data_io import DataIO
from sonogenetics.analysis.analysis_params import dataset_dir, figure_dir_analysis
from utils import load_obj, make_figure, run_job, save_fig
import numpy as np
from typing import List, Any, Dict
import pandas as pd
from pathlib import Path
from sonogenetics.analysis.analyse_responses import BootstrapOutput
from scipy.ndimage import gaussian_filter1d

DEBUG = False

def plot_raster_per_protocol(data_io: DataIO) -> pd.DataFrame:

    cluster_ids = data_io.cluster_df.index.values
    electrodes  = data_io.burst_df.electrode.unique()
    print(f'saving data in: {figure_dir_analysis / "raster plots"}')

    loadname = dataset_dir / f'{data_io.session_id}_cells.csv'
    cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
    pref_ec = detect_preferred_electrode(data_io, cells_df)

    data_io.lock_modification()
    tasks: List[Dict[str, Any]] = []

    for protocol in data_io.burst_df.protocol.unique():

        for cluster_id in cluster_ids:
            for ec in electrodes:

                if ec == pref_ec[protocol].loc[cluster_id, 'ec']:
                    subgroup = 'significant'
                else:
                    subgroup = 'not_selected'

                plotname = f'{cluster_id}_{ec}_{protocol}'

                tasks.append({
                    "data_io": data_io,
                    "cluster_id": cluster_id,
                    "protocol": protocol,
                    "electrode": ec,
                    "savename": (figure_dir_analysis / data_io.session_id / 'raster_plots' / protocol / subgroup / plotname)
                })

    run_job(
        job_fn=single_condition_rasterplot,
        tasks=tasks,
        num_threads=10,
        debug=DEBUG,
    )

def single_condition_rasterplot(data_io: DataIO, cluster_id: str,
                                protocol: str, electrode: str,
                                savename: Path):

    # Setup figure layout
    fig = make_figure(
        width    =1,
        height   =1.5,
        x_domains={
            1: [[0.2, 0.99]],
        },
        y_domains={
            1: [[0.1, 0.9]]
        },
    )

    # Setup variables for plotting
    burst_offset   = 0
    x_plot, y_plot = [], []
    x_lines_laser, y_lines_laser = [], []
    x_lines_dmd, y_lines_dmd = [], []

    yticks         = []
    ytext          = []
    pos            = dict(row=1, col=1)

    d_select = data_io.burst_df.query('electrode == @electrode and '
                                        'protocol == @protocol').copy()
    cluster_data: Dict[str, BootstrapOutput] = load_obj(dataset_dir / 'bootstrapped' / f'bootstrap_{cluster_id}.pkl')

    assert protocol in params_per_protocol.keys(), f'{protocol}'

    params_to_groupby = params_per_protocol[protocol]

    for prm_val, df in d_select.groupby(params_to_groupby):

        train_plot_height_start = burst_offset

        tids = df.train_id.unique()
        assert len(tids) == 1
        tid = tids[0]

        if tid not in cluster_data.keys():
            continue
        trial_data = cluster_data[tid]

        spike_times = trial_data.spike_times
        bins = trial_data.bins

        ystr = ''
        for p, v in zip(params_to_groupby, prm_val):
            ystr += f'{params_abbreviation[p]}: {v:.0f} | '

        ytext.append(ystr)

        yticks.append(burst_offset + len(spike_times) / 2)

        for burst_i, sp in enumerate(spike_times):
            x_plot.append(np.vstack([sp, sp, np.full(sp.size, np.nan)]).T.flatten())
            y_plot.append(np.vstack([np.ones(sp.size) * burst_offset,
                                    np.ones(sp.size)* burst_offset +1, np.full(sp.size, np.nan)]).T.flatten())
            burst_offset += 1

        has_laser = data_io.train_df.loc[tid, 'has_laser']
        bd_laser = data_io.train_df.loc[tid, 'laser_burst_duration']
        if has_laser and bd_laser > 0:
            x_lines_laser.extend([0, bd_laser, bd_laser, 0, 0, None])
            y_lines_laser.extend([train_plot_height_start, train_plot_height_start,
                                  burst_offset, burst_offset, train_plot_height_start, None])

        has_dmd = data_io.train_df.loc[tid, 'has_dmd']
        bd_dmd = data_io.train_df.loc[tid, 'dmd_burst_duration']

        if has_dmd and bd_dmd > 0:
            if has_laser:
                ldelay = data_io.train_df.loc[tid, 'laser_onset_delay']
            else:
                ldelay = 0

            x_lines_dmd.extend([-ldelay, bd_dmd, bd_dmd, -ldelay, -ldelay, None])
            y_lines_dmd.extend([train_plot_height_start, train_plot_height_start,
                                  burst_offset, burst_offset, train_plot_height_start, None])

    if len(x_plot) == 0:
        return

    x_plot = np.hstack(x_plot)
    y_plot = np.hstack(y_plot)

    if has_laser:
        fig.add_scatter(
            x=x_lines_laser, y=y_lines_laser,
            mode='lines', line=dict(width=0.00001, color='black'),
            fill='toself', fillcolor='rgba(200, 50, 50, 0.3)',
            showlegend=False,
            **pos,
        )

    if has_dmd:
        fig.add_scatter(
            x=x_lines_dmd, y=y_lines_dmd,
            mode='lines', line=dict(width=0.00001, color='black'),
            fill='toself', fillcolor='rgba(50, 250, 250, 0.3)',
            showlegend=False,
            **pos,
        )

    fig.add_scatter(
        x = x_plot, y = y_plot,
        mode = 'lines', line = dict(color='black', width=0.5),
        showlegend = False,
        **pos,
    )

    fig.update_xaxes(
        tickvals = np.arange(-500, 500, 100),
        title_text = f'time [ms]',
        range = [bins[0]-1, bins[-1]+1],
        **pos,
    )

    fig.update_yaxes(
        range=[0, burst_offset],
        tickvals = yticks,
        ticktext = ytext,
        **pos,
    )

    save_fig(fig, savename, display=False, verbose=False)


def plot_response_heatmap_per_protocol(data_io: DataIO):
    cluster_ids = data_io.cluster_df.index.values
    electrodes  = data_io.burst_df.electrode.unique()
    print(f'saving data in: {figure_dir_analysis / "raster plots"}')

    loadname = dataset_dir / f'{data_io.session_id}_cells.csv'
    cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
    pref_ec = detect_preferred_electrode(data_io, cells_df)

    data_io.lock_modification()
    tasks: List[Dict[str, Any]] = []

    for protocol in data_io.burst_df.protocol.unique():
        stimsites = data_io.burst_df.query('protocol == @protocol').electrode.unique()

        for cluster_id in cluster_ids:

            ec = pref_ec[protocol].loc[cluster_id, 'ec']

            if ec is None:
                ecs = stimsites
                is_sig = 'not_sig'
            else:
                ecs = [ec]
                is_sig = 'sig'

            for ec in ecs:
                plotname = f'{cluster_id}_{ec}_{protocol}'

                tasks.append({
                    "data_io": data_io,
                    "cluster_id": cluster_id,
                    "electrode": ec,
                    "protocol": protocol,
                    "savename": (figure_dir_analysis / data_io.session_id / 'heatmaps' / protocol / is_sig / plotname)
                })

    run_job(
        job_fn=single_cell_response_heatmap_per_protocol,
        tasks=tasks,
        num_threads=10,
        debug=DEBUG,
    )


def single_cell_response_heatmap_per_protocol(data_io: DataIO,
                                              cluster_id: str,
                                              electrode: str,
                                              protocol: str,
                                              savename: Path):
    # -----------------------------
    # Parameters
    # -----------------------------
    t_pre = 100
    t_after = 200
    stepsize = 5
    binwidth = 30

    smooth_sigma = 3
    baseline_t0 = -120
    baseline_t1 = 0
    max_z = 8


    trials = data_io.train_df.query('protocol == @protocol and electrode == @electrode')

    frates = []
    for ti, tid in enumerate(trials.index.values):

        # Get rename + spike train
        rec_id = trials.loc[tid, 'recording_name']
        spiketrain = data_io.spiketimes[rec_id][cluster_id]

        # Get burs tonsets
        if data_io.train_df.loc[tid, 'has_dmd']:
            burst_onsets = data_io.burst_df.query('train_id == @tid').dmd_burst_onset
        else:
            burst_onsets = data_io.burst_df.query('train_id == @tid').laser_burst_onset

        n_trains = burst_onsets.size

        # Placeholder to extract data into
        bin_centres = np.arange(-t_pre, t_after, stepsize)
        n_bins = bin_centres.size

        # --- Robust baseline indexing ---
        baseline_mask = (bin_centres >= baseline_t0) & (bin_centres < baseline_t1)
        if not np.any(baseline_mask):
            raise ValueError("Baseline window outside bin range.")

        # Get spikes per bin
        binned_sp = np.zeros((n_trains, n_bins))

        for burst_i, burst_onset in enumerate(burst_onsets):
            for bin_i, bin_centre in enumerate(bin_centres):

                # symmetric bin (recommended)
                t0 = burst_onset + bin_centre - binwidth / 2
                t1 = burst_onset + bin_centre + binwidth / 2

                count = np.sum((spiketrain >= t0) & (spiketrain < t1))
                binned_sp[burst_i, bin_i] = count

        # Get mean firing rate
        mean_fr = np.mean(binned_sp, axis=0)
        frates.append(mean_fr)

    cell_fr = np.array(frates)

    # -----------------------------
    # Convert to firing rate (Hz)
    # -----------------------------
    cell_fr = cell_fr / (binwidth / 1000)

    # -----------------------------
    # Optional smoothing
    # -----------------------------
    if smooth_sigma > 0:
        cell_fr = gaussian_filter1d(cell_fr, sigma=smooth_sigma, axis=1)

    # -----------------------------
    # Z-score to baseline window
    # -----------------------------
    baseline = cell_fr[:, baseline_mask]

    baseline_mean = baseline.mean(axis=1, keepdims=True)
    baseline_std = baseline.std(axis=1, keepdims=True)

    # baseline_std[baseline_std == 0] = 1  # avoid divide-by-zero
    idx = np.where(baseline != 0)[0]
    cell_fr[idx] = (cell_fr[idx] - baseline_mean[idx]) / baseline_std[idx]

    # Optional clipping
    cell_fr = np.clip(cell_fr, -max_z, max_z)

    # -----------------------------
    # Plot heatmap
    # -----------------------------
    fig = make_figure(
        height=2,
        y_domains={1: [[0.1, 0.99]]}
    )

    fig.add_heatmap(
        z=cell_fr,
        x=bin_centres,
        y=np.arange(cell_fr.shape[0]),
        colorscale='RdBu_r',
        zmid=0,
        zmin=-max_z,
        zmax=max_z,
        showscale=False
    )

    fig.update_xaxes(
        tickvals=np.arange(-200, 301, 50),
        title_text='time [ms]'
    )

    save_fig(fig=fig, savename=savename, display=False)


if __name__ == '__main__':
    dd = DataIO(dataset_dir)
    session_id = '2026-02-19 mouse c57 5713 Mekano6 A'
    dd.load_session(session_id, load_pickle=True, load_waveforms=False)
    dd.dump_as_pickle()

    plot_raster_per_protocol(dd)
    # plot_response_heatmap_per_protocol(dd)
