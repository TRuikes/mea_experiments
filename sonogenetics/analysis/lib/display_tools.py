from sonogenetics.analysis.lib.analysis_tools import detect_preferred_electrode, get_params_protocol, params_abbreviation
from sonogenetics.analysis.lib.data_io import DataIO
from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis
from utils import load_obj, make_figure, run_job, update_subplot_titles
import numpy as np
from typing import List, Any, Dict
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sonogenetics.project_colors import ProjectColors
from sonogenetics.analysis.lib.bootstrap import BootstrapOutput
import plotly.io as pio
from tqdm import tqdm
from multiprocessing import Pool

DEBUG = True

def write_figure(json_file_path):
    if json_file_path is None:
        return None
    fig = pio.from_json(Path(json_file_path).read_text())
    png_file = Path(json_file_path).with_suffix(".png")
    fig.write_image(
        png_file,
        engine='kaleido',
        scale=4,
        format='png',
    )
    Path(json_file_path).unlink()  # clean up
    return str(png_file)

def generate_raster_plots_session(data_io: DataIO) -> pd.DataFrame:

    print(f'saving data in: {figure_dir_analysis / data_io.session_id / "raster plots"}')

    loadname = dataset_dir / f'{data_io.session_id}_cells.csv'
    cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
    pref_ec = detect_preferred_electrode(data_io, cells_df)

    data_io.lock_modification()
    tasks: List[Dict[str, Any]] = []

    for (rec_id, protocol, ec), df in data_io.train_df.groupby(
            ['rec_id', 'protocol', 'electrode']):

        for cluster_id in data_io.cluster_ids:
            if ec == pref_ec[rec_id][protocol].loc[cluster_id, 'ec']:
                subgroup = 'significant'
            else:
                subgroup = 'not_selected'

            plot_name = f'{cluster_id}_{ec}'
            savename = (figure_dir_analysis / data_io.session_id /
                        'raster_plots' / rec_id / protocol / subgroup / plot_name)

            tasks.append({
                "data_io": data_io,
                "cluster_id": cluster_id,
                "recording_id": rec_id,
                "protocol": protocol,
                "electrode": ec,
                "savename": savename,
            })

    plot_data = run_job(
        job_fn=plot_raster_single_cluster,
        tasks=tasks,
        num_threads=20,
        debug=DEBUG,
    )

    batch_size = 50
    for i in range(0, len(plot_data), batch_size):
        batch = plot_data[i:i+batch_size]
        with Pool(processes=4) as pool:
            # Wrap the iterator with tqdm to show progress
            for _ in tqdm(pool.imap_unordered(write_figure, batch), total=len(batch),
                          desc=f"batch {i//batch_size + 1}"):
                pass  # imap_unordered runs the function and tqdm updates the bar

    data_io.unlock_modification()


def plot_raster_single_cluster(data_io: DataIO,
                               cluster_id: str,
                               recording_id: str,
                               protocol: str,
                               electrode: str,
                               savename: Path,
                              ):

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

    d_select = data_io.burst_df.query(f'electrode == {electrode} and '
                                      f'rec_id == "{recording_id}" and '
                                      f'protocol == "{protocol}"').copy()

    if len(d_select) == 0:
        print(f'cid: {cluster_id}, rid: {recording_id}, ec: {electrode}')
        return
        # raise ValueError('selected dataframe is empty')

    cluster_data: Dict[str, BootstrapOutput] = load_obj(dataset_dir / 'bootstrapped' / f'bootstrap_{cluster_id}.pkl')

    params_to_group_by = get_params_protocol(protocol)

    if 'dac_voltage' in params_to_group_by and 'dac_voltage' not in d_select.columns:
        d_select['dac_voltage'] = d_select['laser_power']

    for prm_val, df in d_select.groupby(params_to_group_by):
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
        for p, v in zip(params_to_group_by, prm_val):
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

        if has_dmd:
            bd_dmd = data_io.train_df.loc[tid, 'dmd_burst_duration']
            if bd_dmd > 0:
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

    # return fig, savename

    # print(f'saved: {savename}')
    # save_fig(fig, savename, display=False, verbose=False)
    # Save as HTML immediately
    json_savename = savename.with_suffix(".json")

    if not json_savename.parent.exists():
        json_savename.parent.mkdir(parents=True, exist_ok=True)
    json_savename.write_text(fig.to_json())
    return str(json_savename)


def generate_heatmaps_session(data_io: DataIO, sig_only=True):
    cluster_ids = data_io.cluster_df.index.values
    print(f'saving data in: {figure_dir_analysis / data_io.session_id / "heatmaps"}')

    loadname = dataset_dir / f'{data_io.session_id}_cells.csv'
    cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
    pref_ec = detect_preferred_electrode(data_io, cells_df)

    data_io.lock_modification()
    tasks: List[Dict[str, Any]] = []

    for (rec_id, protocol, ec), df in data_io.train_df.groupby(['rec_id', 'protocol', 'electrode']):
        for cluster_id in cluster_ids:
            if cluster_id not in pref_ec[rec_id][protocol].index.values:
                continue

            if ec == pref_ec[rec_id][protocol].loc[cluster_id, 'ec']:
                subgroup = 'significant'
            else:
                subgroup = 'not_selected'
                if sig_only:
                    continue

            plot_name = f'{cluster_id}_{ec}'
            savename = (figure_dir_analysis / data_io.session_id / 'heatmaps' /
                        rec_id / protocol / subgroup / plot_name)

            tasks.append({
                "data_io": data_io,
                "cluster_id": cluster_id,
                "electrode": ec,
                "protocol": protocol,
                "savename": savename,
                "recording_id": rec_id,
            })

    print(f'{len(tasks)} tasks')

    figure_files = run_job(
        job_fn=heatmap_per_protocol_slave,
        tasks=tasks,
        num_threads=10,
        debug=DEBUG,
    )

    batch_size = 50
    for i in range(0, len(figure_files), batch_size):
        batch = figure_files[i:i+batch_size]
        with Pool(processes=8) as pool:
            # Wrap the iterator with tqdm to show progress
            for _ in tqdm(pool.imap_unordered(write_figure, batch), total=len(batch),
                          desc=f"batch {i//batch_size + 1}"):
                pass  # imap_unordered runs the function and tqdm updates the bar


def heatmap_per_protocol_slave(data_io: DataIO,
                               cluster_id: str,
                               electrode: str,
                               protocol: str,
                               recording_id: str,
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


    trials = data_io.train_df.query(f'protocol == "{protocol}" and electrode == {electrode}'
                                    f'and rec_id == "{recording_id}"').copy()

    frates = []
    yticks         = []
    ytext          = []
    ytick = 0

    params_to_group_by = get_params_protocol(protocol)

    if 'dac_voltage' in params_to_group_by and 'dac_voltage' not in trials.columns:
        trials['dac_voltage'] = trials['laser_power']

    for prm_val, df in trials.groupby(params_to_group_by):

        if df.shape[0] != 1:
            print('?')
        assert df.shape[0] == 1
        tid = df.iloc[0].train_id

        # Get rename + spike train
        spiketrain = data_io.spiketimes[recording_id][cluster_id]

        # Get burst onsets
        if df.iloc[0]['has_dmd']:
            burst_onsets = data_io.burst_df.query(f'train_id ==  "{tid}"').dmd_burst_onset
        else:
            burst_onsets = data_io.burst_df.query(f'train_id == "{tid}"').laser_burst_onset

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

        ystr = ''
        for p, v in zip(params_to_group_by, prm_val):
            ystr += f'{params_abbreviation[p]}: {v:.0f} | '
        ytext.append(ystr)
        yticks.append(ytick)
        ytick += 1

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
        x_domains={1: [[0.25, 0.99]]},
        y_domains={1: [[0.15, 0.99]]},
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

    fig.update_yaxes(
        tickvals=yticks,
        ticktext=ytext,
    )

    # save_fig(fig=fig, savename=savename, display=False)
    json_savename = savename.with_suffix(".json")

    if not json_savename.parent.exists():
        json_savename.parent.mkdir(parents=True, exist_ok=True)
    json_savename.write_text(fig.to_json())

    # print(f'saved: {json_savename}')
    return str(json_savename)


def firing_rate_per_protocol_master(data_io: DataIO, sig_only=True):
    cluster_ids = data_io.cluster_df.index.values
    print(f'saving data in: {figure_dir_analysis / "firing_rates"}')

    loadname = dataset_dir / f'{data_io.session_id}_cells.csv'
    cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
    pref_ec = detect_preferred_electrode(data_io, cells_df)

    data_io.lock_modification()
    tasks: List[Dict[str, Any]] = []

    for (rec_id, protocol, ec), df in data_io.train_df.groupby(['rec_id', 'protocol', 'electrode']):
        for cluster_id in cluster_ids:
            if cluster_id not in pref_ec[rec_id][protocol].index.values:
                continue

            if ec == pref_ec[rec_id][protocol].loc[cluster_id, 'ec']:
                subgroup = 'significant'
            else:
                subgroup = 'not_selected'
                if sig_only:
                    continue

            plotname = f'{cluster_id}_{ec}_{protocol}'
            savename = (figure_dir_analysis / data_io.session_id / 'firing_rates' /
                        rec_id / protocol / subgroup / plotname)

            tasks.append({
                "data_io": data_io,
                "cluster_id": cluster_id,
                "electrode": ec,
                "protocol": protocol,
                "savename": savename,
            })

    figure_files = run_job(
        job_fn=firing_rate_per_protocol_slave,
        tasks=tasks,
        num_threads=10,
        debug=DEBUG,
    )

    batch_size = 50
    for i in range(0, len(figure_files), batch_size):
        batch = figure_files[i:i+batch_size]
        with Pool(processes=8) as pool:
            # Wrap the iterator with tqdm to show progress
            for _ in tqdm(pool.imap_unordered(write_figure, batch), total=len(batch),
                          desc=f"batch {i//batch_size + 1}"):
                pass  # imap_unordered runs the function and tqdm updates the bar


def firing_rate_per_protocol_slave(data_io: DataIO,
                               cluster_id: str,
                               electrode: str,
                               protocol: str,
                               savename: Path):
    # -----------------------------
    # Parameters
    # -----------------------------

    cluster_data: Dict[str, BootstrapOutput] = load_obj(dataset_dir / 'bootstrapped' / f'bootstrap_{cluster_id}.pkl')


    # t_pre = 100
    # t_after = 200
    # stepsize = 5
    # binwidth = 30

    # smooth_sigma = 3
    # baseline_t0 = -120
    # baseline_t1 = 0

    recording_id = data_io.recording_ids[0]

    clrs = ProjectColors()

    fig_x_domains ={
            1: [[0.02, 0.27], [0.32, 0.57], [0.62, 0.87]],
            2: [[0.02, 0.27], [0.32, 0.57], [0.62, 0.87]],
            3: [[0.02, 0.27], [0.32, 0.57], [0.62, 0.87]],
        }
    fig_y_domains = {
            1: [[0.7, 0.9], [0.7, 0.9], [0.7, 0.9]],
            2: [ [0.4, 0.6], [0.4, 0.6], [0.4, 0.6]],
            3: [[0.1, 0.3], [0.1, 0.3], [0.1, 0.3]],
    }
    fig = make_figure(
        width=1,
        height=2,
        x_domains=fig_x_domains,
        y_domains=fig_y_domains,
        xticks=[], yticks=[],
    )

    # Row 1: PRR 6kHZ, 3 columns for powers, bd in plot
    # Row 2: bd = 20ms, 2 columns for prrs, power in plot
    # Row 3: bd = 20ms, 3 columns for pwrs, prr in plot

    # Plot row 1
    # Configuration for each row
    row_configs = [
        {
            "fixed_param": 'laser_pulse_repetition_rate',
            "groupby": ['train_id', 'dac_voltage', 'laser_burst_duration'],
            "plotrow": 1,
            "col_field": "dac_voltage",
            "clr_field": "laser_burst_duration",
        },
        {
            "fixed_param": 'laser_burst_duration',
            "groupby": ['train_id', 'dac_voltage', 'laser_pulse_repetition_rate'],
            "plotrow": 2,
            "col_field": "laser_pulse_repetition_rate",
            "clr_field": "dac_voltage",
        },
        {
            "fixed_param": 'laser_burst_duration',
            "groupby": ['train_id', 'dac_voltage', 'laser_pulse_repetition_rate'],
            "plotrow": 3,
            "col_field": "dac_voltage",
            "clr_field": "laser_pulse_repetition_rate",
        },
    ]

    subplot_titles = {}
    r_min, r_max = {}, {}

    if 'dac_voltage' not in data_io.train_df.columns:
        data_io.train_df['dac_voltage'] = data_io.train_df['laser_power']

    for cfg in row_configs:

        plotrow = cfg["plotrow"]

        # Find the fixed parameter and its fixed values
        fixed_name = cfg['fixed_param']
        fixed_val = data_io.train_df[fixed_name].max()

        tdf = data_io.train_df.query(
            f'rec_id == "{recording_id}" and '
            f'protocol == "{protocol}" and '
            f'electrode == {electrode} and '
            f'{fixed_name} == {fixed_val}')

        # Find the parameter to spread over the columns and its values
        col_name = cfg['col_field']
        col_values = tdf[col_name].unique()

        if len(col_values) > 3:
            col_values = np.sort(col_values[-3:])

        for col_i, col_value in enumerate(col_values):
            cdf = tdf.query(f'{col_name} == {col_value}')

            # Find the parameter and values to spread voer the colors in this subplot
            clr_name = cfg['clr_field']
            clr_values = cdf[clr_name].unique()
            pos = dict(row=plotrow, col=col_i + 1)

            skey = (plotrow, col_i + 1)
            if skey not in subplot_titles.keys():
                if col_i == 0:
                    txt = f'{params_abbreviation[fixed_name]}: {fixed_val:.0f} | '
                else:
                    txt = ''
                txt += f'{params_abbreviation[col_name]}: {col_value:.0f}'

                subplot_titles[skey] = txt

            for clr_value in clr_values:
                plot_df = cdf.query(f'{clr_name} == {clr_value}')
                assert plot_df.shape[0] == 1


                train_id = plot_df.iloc[0].train_id


                if train_id not in cluster_data.keys():
                    continue

                bins = cluster_data[train_id].get('bins')
                firing_rate = cluster_data[train_id].get('firing_rate')
                # firing_rate_ci_low = cluster_data[tid].get('firing_rate_ci_low')
                # firing_rate_ci_high = cluster_data[tid].get('firing_rate_ci_high')

                plot_id = (plotrow, col_i + 1)
                if plot_id not in r_min.keys():
                    r_min[plot_id] = np.min(firing_rate)
                else:
                    if np.min(firing_rate) < r_min[plot_id]:
                        r_min[plot_id] = np.min(firing_rate)

                if plot_id not in r_max.keys():
                    r_max[plot_id] = np.max(firing_rate)
                else:
                    if np.max(firing_rate) > r_max[plot_id]:
                        r_max[plot_id] = np.max(firing_rate)

                if cfg["clr_field"] == 'laser_burst_duration':
                    clr = clrs.burst_duration(clr_value)
                elif cfg["clr_field"] == 'laser_pulse_repetition_rate':
                    clr = clrs.laser_prr(clr_value)
                elif cfg["clr_field"] == 'dac_voltage':
                    clr =clrs.min_max_map(val=clr_value, min_val=2000, max_val=8000)
                else:
                    raise ValueError('')

                fig.add_scatter(
                    x=bins,
                    y=firing_rate,
                    mode='lines',
                    name=f'{params_abbreviation[clr_name]}: {clr_value:.0f}',
                    showlegend=True if col_i == 0 else False,
                    legendgroup=f'{pos["row"]}',
                    line=dict(color=clr, width=1.5),
                    **pos,
                )

    if len(r_min.keys()) == 0:
        return None

    row_positions = [
        (row, col)
        for col in range(1, 4) for row in range(1, 4)
        if (row, col) in r_min.keys()
    ]

    row_min = min(r_min[pos] for pos in row_positions)
    row_max = max(r_max[pos] for pos in row_positions)

    dy = row_max - row_min
    ymin = row_min - 0.1 * dy
    ymax = row_max + 0.1 * dy

    for row in range(1, 4):
        for col in range(1, 4):

            fig.add_scatter(
                x=[0, 0],
                y=[ymin, ymax],
                mode='lines', line=dict(color='black', width=0.5, dash='2px'),
                row=row, col=col,
                showlegend=False,
            )

            fig.update_yaxes(
                range=[ymin, ymax],
                row=row, col=col,
                tickvals=np.arange(0, 100, 20),
                ticktext=np.arange(0, 100, 20) if col == 1 else ['' for _ in range(10)],
                ticklen=1.5, tickwidth=0.5,
            )

            fig.update_xaxes(
                tickvals=np.arange(-200, 500, 50),
                range=[-100, 200],
                row=row, col=col,
                ticklen=1.5, tickwidth=0.5,
                ticktext=np.arange(-200, 500, 50),

            )


    fig.update_layout(
        legend=dict(
            xanchor='left',
            x=0.85,
            yanchor='middle',
            y=0.5,
            xref='paper',
            entrywidth=0.1,
            itemwidth=30,
        )
    )

    fig = update_subplot_titles(fig,
                                x_domains=fig_x_domains,
                                y_domains=fig_y_domains,
                                subplot_titles=subplot_titles,)

    # save_fig(fig=fig, savename=savename, display=False)
    json_savename = savename.with_suffix(".json")

    if not json_savename.parent.exists():
        json_savename.parent.mkdir(parents=True, exist_ok=True)
    json_savename.write_text(fig.to_json())

    # print(f'saved: {json_savename}')
    return str(json_savename)



if __name__ == '__main__':
    dd = DataIO(dataset_dir)
    session_id = '2026-02-19 mouse c57 5713 Mekano6 A'
    dd.load_session(session_id, load_pickle=True, load_waveforms=False)
    # dd.dump_as_pickle()

    # plot_raster_per_protocol(dd)
    # plot_response_heatmap_per_protocol(dd)
    res = firing_rate_per_protocol_slave(data_io=dd, cluster_id=dd.cluster_ids[3], electrode=47, protocol='pa_dose_sequence_1',
                                   savename=figure_dir_analysis / 'test')
    write_figure(res)
