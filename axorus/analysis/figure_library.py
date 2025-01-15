import utils
import numpy as np


def plot_session_waveforms(data_io, savedir, rec_nrs):
    """
    write all cluster waveforms into savedir, select waveforms
    from recordingnumbers set in rec nrs

    :param data_io: preloaded dataset object
    :param savedir: directory to save figures in
    :param rec_nrs: recording numbers (2) to draw waveforms from
    :return:
    """
    print(f'Loading cluster waveforms:')

    waveform_figures = []
    recnames = list(data_io.waveforms.keys())

    # Make figure per cluster
    for cluster_id, cinfo in data_io.cluster_df.iterrows():

        # Create figur eobject
        fig = utils.make_figure(
            width=1,
            height=1,
            x_domains={
                1: [[0.1, 0.4], [0.6, 0.9]],
            },
            y_domains={
                1: [[0.1, 0.9], [0.1, 0.9]]
            },
            subplot_titles={
                1: [f'{cluster_id}', ''],
            }
        )

        # Load data per recording
        for ri, rid in enumerate([recnames[rec_nrs[0]], recnames[rec_nrs[1]]]):

            # Load this clusters' waveform for current recording name
            wv = data_io.waveforms[rid][cluster_id]

            if wv.size == 0:  # If no waveforms are available continue
                continue

            # Assign subplot where to plot it
            pos = dict(row=1, col=ri+1)

            # Gather X and Y traces to plot
            x_plot = []
            y_plot = []
            n_pts = wv.shape[1]
            for wave_i in range(wv.shape[0]):
                x_plot.append(np.arange(0, n_pts, 1) / 20)
                x_plot.append([None])

                y = wv[wave_i, :] - np.mean(wv[wave_i, :])
                y_plot.append(y)
                y_plot.append([None])
            x_plot = np.hstack(x_plot)
            y_plot = np.hstack(y_plot)

            # Plot the waveforms
            fig.add_scatter(
                x=x_plot, y=y_plot,
                mode='lines', line=dict(color='black', width=0.1),
                showlegend=False,
                **pos,
            )

            # Style figure x-axes
            fig.update_xaxes(
                tickvals=np.arange(0, 30, 5),
                title_text='time [ms]'
            )
            # Style figure y-axes
            fig.update_yaxes(
                range=[-250, 200],
                tickvals=np.arange(-400, 400, 100),
                **pos)

        # Write figure to drive
        savename = savedir / cluster_id
        utils.save_fig(fig, savename, display=False)
        waveform_figures.append(savename.as_posix() + '.png')

    # Create powerpoint slides with all waveforms
    utils.create_ppt_with_figures_2_row_1_col(waveform_figures, savedir / 'waveforms.pptx')

#
# def plot_single_cell_response_curves():
#     electrodes = burst_df.electrode.unique()
#     n_rows = 3
#     n_cols = int(np.ceil(len(electrodes) / n_rows))
#
#     x_domains = {}
#     y_domains = {}
#     subplot_titles = {}
#     x_offset = 0.05
#     y_offset = 0.1
#     x_spacing = 0.1
#     y_spacing = 0.1
#
#     x_width = (1 - (n_cols - 1) * x_spacing - 2 * x_offset) / n_cols
#     y_height = (1 - (n_rows - 1) * y_spacing - 2 * y_offset) / n_rows
#
#     electrode_i = 0
#     electrode_pos = {}
#
#     for row_i in range(n_rows):
#         x_domains[row_i + 1] = []
#         y_domains[row_i + 1] = []
#         subplot_titles[row_i + 1] = []
#
#         for col_j in range(n_cols):
#             x0 = x_offset + col_j * (x_spacing + x_width)
#             x_domains[row_i + 1].append([x0, x0 + x_width])
#
#             y1 = 1 - y_offset - row_i * (y_spacing + y_height)
#             y_domains[row_i + 1].append([y1 - y_height, y1])
#
#             if electrode_i >= len(electrodes):
#                 subplot_titles[row_i + 1].append(f'')
#             else:
#                 subplot_titles[row_i + 1].append(f'{electrodes[electrode_i]:.0f}')
#                 electrode_pos[electrodes[electrode_i]] = dict(row=row_i + 1, col=col_j + 1)
#             electrode_i += 1
#
#     for cluster_id in data_io.cluster_df.index.values:
#         # Find distance from laser
#         cx = data_io.cluster_df.loc[cluster_id].cluster_x
#         cy = data_io.cluster_df.loc[cluster_id].cluster_y
#
#         for electrode in electrode_pos.keys():
#             r = electrode_pos[electrode]['row']
#             c = electrode_pos[electrode]['col']
#
#             lx = data_io.burst_df.query(f'electrode == {electrode}').iloc[0].laser_x
#             ly = data_io.burst_df.query(f'electrode == {electrode}').iloc[0].laser_y
#
#             d = np.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)
#
#             subplot_titles[r][c - 1] = f'{electrode:.0f} - d = {d:.0f} um'
#
#         fig = utils.make_figure(
#             width=1, height=1.2,
#             x_domains=x_domains,
#             y_domains=y_domains,
#             subplot_titles=subplot_titles,
#         )
#
#         fr_max_all = 0
#
#         for electrode, df in burst_df.groupby('electrode'):
#             pos = electrode_pos[electrode]
#
#             n_pulses = []
#             fr_baseline = []
#             fr_response = []
#             xticks = []
#
#             for train_id, tdf in df.sort_values('duty_cycle').groupby('train_id'):
#                 rf = tdf.iloc[0].repetition_frequency / 1000
#                 bd = tdf.iloc[0].burst_duration
#
#                 n_pulses.append(rf * bd)
#                 xticks.append(f'{rf * bd:.0f}')
#
#                 # rf.append(f'{tdf.iloc[0].repetition_frequency/1000:.1f}')
#                 # xticks.append(f'{tdf.iloc[0].repetition_frequency/1000:.1f}')
#
#                 fr_baseline.append(cells_df.loc[cluster_id, (train_id, 'baseline_firing_rate')])
#                 fr_response.append(cells_df.loc[cluster_id, (train_id, 'response_firing_rate')])
#
#             # convert lists to numpy arrays
#             n_pulses = np.array(n_pulses)
#             fr_baseline = np.array(fr_baseline)
#             fr_response = np.array(fr_response)
#             xticks = np.array(xticks)
#
#             # sort by dc values
#             sort_idx = np.argsort(n_pulses)
#             fr_baseline = fr_baseline[sort_idx]
#             fr_response = fr_response[sort_idx]
#             n_pulses = n_pulses[sort_idx]
#             xticks = xticks[sort_idx]
#
#             # Plot baseline data
#             fig.add_scatter(
#                 x=n_pulses, y=fr_baseline,
#                 mode='lines+markers',
#                 line=dict(color='grey', width=1),
#                 showlegend=True if pos['row'] == 1 and pos['col'] == 1 else False,
#                 name='baseline',
#                 **pos,
#             )
#
#             # Plot response data
#             fig.add_scatter(
#                 x=n_pulses, y=fr_response,
#                 mode='lines+markers',
#                 line=dict(color='darkgreen', width=1),
#                 showlegend=True if pos['row'] == 1 and pos['col'] == 1 else False,
#                 name='response',
#                 **pos,
#             )
#
#             if np.all(pd.isna(fr_baseline)) and np.all(pd.isna(fr_response)):
#                 continue
#
#             fr_max = np.nanmax([np.nanmax(fr_baseline), np.nanmax(fr_response)])
#             fr_max_all = max(fr_max, fr_max_all)
#
#         for col in range(n_cols):
#             fig.update_xaxes(
#                 tickvals=xticks,
#                 title_text='n pulses in 10 ms',
#                 row=n_rows,
#                 col=col + 1,
#             )
#
#         for electrode in electrode_pos.keys():
#             row = electrode_pos[electrode]['row']
#             col = electrode_pos[electrode]['col']
#
#             if fr_max_all < 10:
#                 tickvals = np.arange(0, 12, 1)
#                 r = [0, 10]
#             elif fr_max_all < 20:
#                 tickvals = np.arange(0, 22, 5)
#                 r = [0, 20]
#             elif fr_max_all < 60:
#                 tickvals = np.arange(0, 62, 20)
#                 r = [0, 60]
#             elif fr_max_all < 100:
#                 tickvals = np.arange(0, 101, 50)
#                 r = [0, 101]
#             elif fr_max_all < 200:
#                 tickvals = np.arange(0, 201, 50)
#                 r = [0, 201]
#             else:
#                 tickvals = np.arange(0, 501, 100)
#                 r = [0, 501]
#
#             fig.update_yaxes(
#                 tickvals=tickvals,
#                 range=r,
#                 row=row, col=col,
#             )
#
#         sname = figure_dir / session_id / 'pa_dc_min_max' / 'curve_per_cell' / cluster_id
#         utils.save_fig(fig, sname, display=False)


def get_heatmap_relative_to_laser(data_io, cells_df, train_ids, binsize=30, significant_only=False):


    # Define range of the heatmap
    range_min, range_max = -300, 300

    # Construct x and y bins
    xbins = np.arange(range_min, range_max + binsize, binsize)
    ybins = np.arange(range_min, range_max + binsize, binsize)


    # Create maps to store data in
    fr_sum = np.zeros((xbins.size, ybins.size))
    fr_count = np.zeros((xbins.size, ybins.size))
    fr_map = np.zeros((xbins.size, ybins.size))


    for train_id in train_ids:
        # Find train information; laser position
        bursts = data_io.burst_df.query(f'train_id == "{train_id}"')
        laser_x = bursts.iloc[0].laser_x
        laser_y = bursts.iloc[0].laser_y

        # Load data per cluster
        for i, r in data_io.cluster_df.iterrows():

            # Find cluster position
            lx = r.cluster_x
            ly = r.cluster_y

            # Find cluster position relative to laser
            x_rel = lx - laser_x
            y_rel = ly - laser_y

            # Find index into heatmap
            xi = np.searchsorted(xbins, x_rel, side='right') - 1
            yi = np.searchsorted(ybins, y_rel, side='right') - 1

            # If no data available for cluster, skip it
            if i not in cells_df.index.values:
                continue

            # Extract stats for this trial
            fr = cells_df.loc[i, (train_id, 'response_firing_rate')]
            is_sig = cells_df.loc[i, (train_id, 'is_significant')]

            if significant_only and not is_sig: continue

            # Assign this clusters' firing rate to this bin
            fr_count[xi, yi] += 1
            fr_sum[xi, yi] += fr

    # Find the average firing rate
    fr_average = fr_sum / fr_count

    return xbins, ybins, fr_average



if __name__ == '__main__':
    from pathlib import Path
    from axorus.data_io import DataIO
    import pandas as pd
    from axorus.preprocessing.project_colors import ProjectColors


    # Load data
    session_id = '241213_A'
    data_dir = Path(r'E:\Axorus\ex_vivo_series_3\dataset')
    figure_dir = Path(r'C:\Axorus\figures')
    data_io = DataIO(data_dir)
    loadname = data_dir / f'{session_id}_cells.csv'
    data_io.load_session(session_id)
    cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
    clrs = ProjectColors()

    INCLUDE_RANGE = 50  # include cells at max distance = 50 um

    train_id = data_io.burst_df.query('protocol == "pa_dc_min_max_series"').train_id.unique()[0]

    get_trial_heatmap(data_io, cells_df, train_id)