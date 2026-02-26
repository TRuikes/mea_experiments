from sonogenetics.analysis.analysis_tools import detect_preferred_electrode, params_per_protocol, params_abbreviation
from sonogenetics.analysis.data_io import DataIO
from sonogenetics.analysis.analysis_params import dataset_dir, figure_dir_analysis
from utils import load_obj, make_figure, run_job, save_fig
import numpy as np
from typing import List, Any, Dict
import pandas as pd
from pathlib import Path
DEBUG = True

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

                if ec == pref_ec.loc[cluster_id, 'ec']:
                    subgroup = 'significant'
                else:
                    subgroup = 'not_selected'

                plotname = f'{cluster_id}_{ec}_{protocol}'

                tasks.append({
                    "data_io": data_io,
                    "cluster_id": cluster_id,
                    "protocol": protocol,
                    "electrode": ec,
                    "savename": (figure_dir_analysis / data_io.session_id / 'raster_plots' / subgroup / plotname)
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
    cluster_data = load_obj(dataset_dir / 'bootstrapped' / f'bootstrap_{cluster_id}.pkl')

    assert protocol in params_per_protocol.keys(), f'{protocol}'

    params_to_groupby = params_per_protocol[protocol]

    for prm_val, df in d_select.groupby(params_to_groupby):

        train_plot_height_start = burst_offset

        tids = df.train_id.unique()
        assert len(tids) == 1
        tid = tids[0]
        spike_times = cluster_data[tid]['spike_times']
        bins = cluster_data[tid]['bins']

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

        has_laser = data_io.burst_df.query('train_id == @tid').iloc[0].has_laser
        bd_laser = data_io.burst_df.query('train_id == @tid').iloc[0].laser_burst_duration
        if has_laser and bd_laser > 0:
            x_lines_laser.extend([0, bd_laser, bd_laser, 0, 0, None])
            y_lines_laser.extend([train_plot_height_start, train_plot_height_start,
                                  burst_offset, burst_offset, train_plot_height_start, None])

        bd_dmd = data_io.burst_df.query('train_id == @tid').iloc[0].dmd_burst_duration
        has_dmd = data_io.burst_df.query('train_id == @tid').iloc[0].has_dmd

        if has_dmd and bd_dmd > 0:
            if has_laser:
                ldelay = data_io.burst_df.query('train_id == @tid').iloc[0].laser_onset_delay
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


if __name__ == '__main__':
    dd = DataIO(dataset_dir)
    session_id = '2026-02-19 mouse c57 5713 Mekano6 A'

    figure_dir_analysis = figure_dir_analysis / session_id
    dd.load_session(session_id, load_pickle=False, load_waveforms=False)
    dd.dump_as_pickle()

    plot_raster_per_protocol(dd)
