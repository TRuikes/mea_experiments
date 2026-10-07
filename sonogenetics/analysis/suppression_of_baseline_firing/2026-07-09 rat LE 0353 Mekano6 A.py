import pandas as pd
import numpy as np
import utils
from sonogenetics.analysis.lib.data_io import get_dataio, DataIO
from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis
from utils import load_obj
from typing import Dict
from sonogenetics.analysis.lib.bootstrap import BootstrapOutput
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Polygon


LASER_ONSET_DELAY = 40

def main():
    data_io = get_dataio()
    data_io.load_session('2026-07-09 rat LE 0353 Mekano6 A')
    rec_id = "rec_2_A_20260709_pa_dmd_timing_full_field"
    tasks = []

    for LASER_ONSET_DELAY in [0, 40, 60]:
        trials_df = data_io.train_df.query(''
                                           'laser_power == 5000 and '
                                           f'laser_onset_delay == {LASER_ONSET_DELAY} and '
                                           f'rec_id == "{rec_id}"')

        clusters = data_io.cluster_ids

        for ec, tdf in trials_df.groupby('electrode'):
            for cid in clusters:
                savename = figure_dir_analysis / data_io.session_id / 'suppression_of_baseline_firing' / f'{ec:.0f}' / f'{LASER_ONSET_DELAY:.0f}' / cid


                tids = list(tdf.index.values)
                tdf_dmd_only = data_io.train_df.query('has_dmd == 1 and has_laser == 0 and '
                                             f'rec_id == "{rec_id}" and electrode == {ec}')
                tdf_laser_only = data_io.train_df.query('has_dmd == 0 and has_laser == 1 and '
                                                        f'rec_id == "{rec_id}" and electrode == {ec} and '
                                                        f'laser_burst_duration == 100 and laser_power == 5000')
                tids.extend(tdf_dmd_only.index.values)
                tids.extend(tdf_laser_only.index.values)

                tasks.append({
                    'data_io': data_io,
                    'trial_ids': tids,
                    'cluster_id': cid,
                    'savename': savename,
                })

    utils.run_job(
        job_fn=plot_raster_single_cluster,
        tasks=tasks,
        num_threads=10,
        debug=True,
    )


def plot_raster_single_cluster(data_io: DataIO,
                               trial_ids: list,
                               cluster_id: str,
                               savename: Path,
                               ):

    # Setup variables for plotting
    burst_offset = 0
    x_plot, y_plot = [], []
    x_lines_laser, y_lines_laser = [], []
    x_lines_dmd, y_lines_dmd = [], []

    yticks = []
    ytext = []


    cluster_data: Dict[str, BootstrapOutput] = load_obj(
        dataset_dir / 'bootstrapped' / f'bootstrap_{cluster_id}.pkl')

    for tid in trial_ids:

        train_plot_height_start = burst_offset

        if tid not in cluster_data.keys():
            continue

        trial_data = cluster_data[tid]
        if trial_data is None:
            continue

        spike_times = trial_data.spike_times
        bins = trial_data.bins

        # ytext.append(ystr)

        yticks.append(burst_offset + len(spike_times) / 2)

        for burst_i, sp in enumerate(spike_times):
            sp_rel = sp - LASER_ONSET_DELAY
            x_plot.append(np.vstack([sp_rel, sp_rel, np.full(sp_rel.size, np.nan)]).T.flatten())
            y_plot.append(np.vstack([np.ones(sp_rel.size) * burst_offset,
                                     np.ones(sp_rel.size) * burst_offset + 1,
                                     np.full(sp_rel.size, np.nan)]).T.flatten())
            burst_offset += 1

        has_laser = data_io.train_df.loc[tid, 'has_laser']
        has_dmd = data_io.train_df.loc[tid, 'has_dmd']

        if has_laser:
            if has_dmd:
                laser_onset_delay = data_io.train_df.loc[tid, 'laser_onset_delay'] - LASER_ONSET_DELAY
            else:
                laser_onset_delay = 0
            laser_burst_duration = data_io.train_df.loc[tid, 'laser_burst_duration']
        else:
            laser_onset_delay, laser_burst_duration = None, None

        if has_dmd:
            dmd_onset_delay_base = data_io.train_df.loc[tid, 'dmd_onset_delay']
            dmd_onset_delay = data_io.train_df.loc[tid, 'dmd_onset_delay'] - LASER_ONSET_DELAY
            dmd_burst_duration = data_io.train_df.loc[tid, 'dmd_burst_duration']

        else:
            dmd_onset_delay = None
            dmd_onset_delay_base = None
            dmd_burst_duration = None


        # Shared Y-coordinates for the bounding boxes
        y_box = [train_plot_height_start, train_plot_height_start, burst_offset, burst_offset,
                 train_plot_height_start,
                 None]

        # 1. Handle DMD shading
        if has_dmd:
            x_lines_dmd.extend([dmd_onset_delay, dmd_onset_delay+dmd_burst_duration,
                                dmd_onset_delay+dmd_burst_duration, dmd_onset_delay, dmd_onset_delay, None])
            y_lines_dmd.extend(y_box)

        # 2. Handle Laser shading (calculate alignment shift automatically)
        if has_dmd and has_laser:
            laser_shift = (laser_onset_delay - dmd_onset_delay_base)
        else:
            laser_shift = 0
        # assert laser_shift == 0, laser_onset_delay

        if has_laser:
            x_lines_laser.extend([laser_shift, laser_shift + laser_burst_duration,
                                  laser_shift + laser_burst_duration, laser_shift, laser_shift, None])
            y_lines_laser.extend(y_box)

    if len(x_plot) == 0:
        return

    x_plot = np.hstack(x_plot)
    y_plot = np.hstack(y_plot)

    # Setup figure (roughly matching the original width=1(unit)x height=1.5(unit) aspect,
    # scaled up to a reasonable pixel size)
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.subplots_adjust(left=0.4, right=0.99, bottom=0.1, top=0.9)

    # Laser shading (drawn as filled rectangles, one per trial)
    for seg_x, seg_y in _split_on_none(x_lines_laser, y_lines_laser):
        poly = Polygon(
            np.column_stack([seg_x, seg_y]),
            closed=True,
            facecolor=(200 / 255, 50 / 255, 50 / 255, 0.3),
            edgecolor='none',
            zorder=1,
        )
        ax.add_patch(poly)

    # DMD shading (drawn as filled rectangles, one per trial)
    for seg_x, seg_y in _split_on_none(x_lines_dmd, y_lines_dmd):
        poly = Polygon(
            np.column_stack([seg_x, seg_y]),
            closed=True,
            facecolor=(50 / 255, 250 / 255, 250 / 255, 0.3),
            edgecolor='none',
            zorder=1,
        )
        ax.add_patch(poly)

    # Spike raster (matplotlib respects NaN as a line-break, same as plotly)
    ax.plot(x_plot, y_plot, color='black', linewidth=0.5, zorder=2)

    # X-axis
    ax.set_xlim(-100, 200 + 1)
    ax.set_xticks(np.arange(-100, 201, 100))
    ax.set_xlabel('time [ms]')

    # Y-axis
    ax.set_ylim(0, burst_offset)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytext)

    savename = Path(savename)
    if not savename.parent.exists():
        savename.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(savename, dpi=200)
    plt.close(fig)


def _split_on_none(xs, ys):
    """Split flat (x, y) lists containing None separators into a list of
    (x_segment, y_segment) polygon/line pieces."""
    segments = []
    cur_x, cur_y = [], []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            if cur_x:
                segments.append((cur_x, cur_y))
            cur_x, cur_y = [], []
        else:
            cur_x.append(x)
            cur_y.append(y)
    if cur_x:
        segments.append((cur_x, cur_y))
    return segments


if __name__ == "__main__":
    main()