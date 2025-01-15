import pandas as pd
from pathlib import Path
from axorus.data_io import DataIO
import utils
import numpy as np
from axorus.preprocessing.project_colors import ProjectColors

# Load project colors
clrs = ProjectColors()

# Define paths for input and output
data_dir = Path(r'C:\axorus\dataset')
figure_dir = Path(r'C:\Axorus\figures')

# Define sessions to plot
sessions = (
    # '241024_A',  # has high repetition freqs at multiple laser levels
    '241108_A',
    '241211_A',
    '241213_A'
)


# Define a function to print stats for each session
# and detect what cells to include
def print_stats():
    data_io = DataIO(data_dir)

    # Create placeholder for cells to include
    cell_overview = pd.DataFrame()
    row_i = 0

    # Print info per sessions
    for sid in sessions:

        # Print session name
        print(f'\n\n{sid}')
        data_io.load_session(sid, load_waveforms=False)
        cells_df = pd.read_csv(data_dir / f'{sid}_cells.csv', header=[0, 1], index_col=0)

        # Get burst_info for pa min max series
        burst_df = data_io.burst_df.query('protocol == "pa_dc_min_max_series"')

        for prf, prf_df in burst_df.groupby('repetition_frequency'):

            irradiances = prf_df.irradiance_large_fiber_diameter.unique()
            e_pulses = prf_df.e_pulse.unique()
            fiber_diameter = prf_df.fiber_diameter.unique()[0] * 1000
            assert len(irradiances) == 1
            print(f'{prf:.0f} | irradiance: {irradiances[0]:.0f} W/mm2 | e pulse: {e_pulses[0]:.2f} uJ'
                  f'| fiber: {fiber_diameter:.0f} um')

        # Detect what cells to include
        # Cells are only included if they are significantly responding
        # at maximum repetition frequency in this session
        max_rpf = burst_df.repetition_frequency.max()
        train_ids = burst_df.query(f'repetition_frequency == {max_rpf}').train_id.unique()

        for train_id in train_ids:
            # Check per cell
            for cid in data_io.cluster_df.index.tolist():


                # Do not include, if no response at max prf
                if not cells_df.loc[cid, (train_id, 'is_significant')] or pd.isna(cells_df.loc[cid, (train_id, 'is_significant')]):
                    continue

                cell_overview.at[row_i, 'cluster_id'] = cid
                cell_overview.at[row_i, 'train_id'] = train_id
                cell_overview.at[row_i, 'session_id'] = sid

                # Detect at which electrode the current train was
                electrode = burst_df.query('train_id == @train_id').electrode.values[0]
                cell_overview.at[row_i, 'electrode'] = electrode

                if electrode == 156 and cid == 'uid_081124_083':
                    print('x')

                row_i += 1

    return cell_overview


def plot_cell_count_vs_distance_laser():

    # Load dataset
    data_io = DataIO(data_dir)

    # Setup placeholder for data
    cell_distance_vs_laser = pd.DataFrame()
    row_i = 0

    for sid in sessions:
        print(f'\n\n{sid}')

        # Load session data
        data_io.load_session(sid, load_waveforms=False)

        # Get burst_info for pa min max series
        burst_df = data_io.burst_df.query('protocol == "pa_dc_min_max_series"')

        # Get train ids
        train_ids = burst_df.train_id.unique()

        for train_id in train_ids:
            trial_info = data_io.burst_df.query(f'train_id == "{train_id}"').iloc[0]

            for cid in data_io.cluster_df.index.tolist():

                # Find distance to laser for this cluster in this trial
                laser_x, laser_y = trial_info.laser_x, trial_info.laser_y
                cluster_x, cluster_y = data_io.cluster_df.loc[cid, 'cluster_x'], data_io.cluster_df.loc[cid, 'cluster_y']
                cell_distance_vs_laser.at[row_i, 'd'] = np.sqrt((cluster_x - laser_x)**2 + (cluster_y - laser_y)**2)
                row_i += 1


    # Create figure
    fig = utils.simple_fig(
        width=1, height=1
    )

    fig.add_histogram(
        x=cell_distance_vs_laser['d'].values,
        xbins=dict(size=30)
    )

    fig.update_xaxes(
        tickvals=np.arange(0, 600, 100),
        title_text=f'd [um]'
    )

    fig.update_yaxes(
        title_text='n rec clusters'
    )

    sname = figure_dir / 'pa_dc_min_max' / 'distance_distribution'
    utils.save_fig(fig, sname, display=True)




def plot_cell_firing_rate_vs_distance_per_prf(cell_overview):

    # Setup bins for binning the distances
    d_max = 500
    d_width = 30

    # Create bins
    bins = np.arange(0, d_max + d_width, d_width)
    # Compute bin centers
    bin_centers = bins[:-1] + d_width / 2  # Exclude the last bin edge and calculate centers

    # Load dataset
    data_io = DataIO(data_dir)

    # Create placeholders for firing rates
    df = pd.DataFrame()
    df_idx = 0

    # Load data per session, into the placeholder dataframe
    for sid in sessions:

        # Print session name
        print(f'\n\nplotting {sid}')
        data_io.load_session(sid, load_waveforms=False)
        cells_df = pd.read_csv(data_dir / f'{sid}_cells.csv', header=[0, 1], index_col=0)

        # Get burst_info for pa min max series
        burst_df = data_io.burst_df.query('protocol == "pa_dc_min_max_series"')

        # Detect repetition frequencies
        prfs = burst_df.repetition_frequency.unique()

        # Get cells to include for this session
        co = cell_overview.query(f'session_id == "{sid}"')

        for (prf, electrode), prf_df in burst_df.groupby(['repetition_frequency', 'electrode']):

            for i, r in co.iterrows():

                trial_info = prf_df.iloc[0]
                train_id = trial_info.train_id

                # Find distance to laser for this cluster in this trial
                laser_x, laser_y = trial_info.laser_x, trial_info.laser_y
                cluster_x, cluster_y = data_io.cluster_df.loc[r.cluster_id, 'cluster_x'], data_io.cluster_df.loc[r.cluster_id, 'cluster_y']
                df.at[df_idx, 'd'] = np.sqrt((cluster_x - laser_x)**2 + (cluster_y - laser_y)**2)
                df.at[df_idx, 'fr'] = cells_df.loc[r.cluster_id, (train_id, 'response_firing_rate')]
                df.at[df_idx, 'lat'] = cells_df.loc[r.cluster_id, (train_id, 'response_latency')]
                df.at[df_idx, 'prf'] = prf
                df_idx += 1

    df['bin'] = pd.cut(df['d'], bins=bins, right=False, include_lowest=True)
    df['prf_bin'] = pd.cut(df['prf'],
                           bins=[1000, 1900, 2400, 3500, 4000, 4600, 5000, 6000, 7200],
                           right=False, include_lowest=True)

    # Create figure
    fig = utils.simple_fig(
        width=1, height=1
    )

    for prf, prf_df in df.groupby('prf_bin'):
        # Compute mean values of 'fr' for each bin
        mean_values = prf_df.groupby('bin')['fr'].mean().values

        # Plot data
        fig.add_scatter(
            x=bin_centers, y=mean_values,
            mode='lines+markers', line=dict(color=clrs.repetition_frequency(np.mean([prf.left, prf.right]))),
            showlegend=False,
            # name=f'{n_pulses:.0f}',
        )

    fig.update_xaxes(
        title_text='d laser [um]',
        tickvals=np.arange(0, 600, 100),
    )
    fig.update_yaxes(
        title_text='firing rate [Hz]',
        tickvals=np.arange(0, 200, 50),
    )

    sname = figure_dir / 'pa_dc_min_max' / 'response_firing_rate_vs_distance'
    utils.save_fig(fig, sname, display=True)

    # Create figure
    fig = utils.simple_fig(
        width=1, height=1
    )

    for prf, prf_df in df.groupby('prf_bin'):
        # Compute mean values of 'latency' for each bin
        mean_values = prf_df.groupby('bin')['lat'].mean().values

        # Plot data
        fig.add_scatter(
            x=bin_centers, y=mean_values,
            mode='lines+markers', line=dict(color=clrs.repetition_frequency(np.mean([prf.left, prf.right]))),
            showlegend=False,
            # name=f'{n_pulses:.0f}',
        )

    fig.update_xaxes(
        title_text='d laser [um]',
        tickvals=np.arange(0, 600, 100),
    )
    fig.update_yaxes(
        title_text='response latency [ms]',
        tickvals=np.arange(0, 250, 50),
        range=[0, 201],
    )

    sname = figure_dir / 'pa_dc_min_max' / 'response_latency_vs_distance'
    utils.save_fig(fig, sname, display=True)


def plot_percent_modulated_cells():
    data_io = DataIO(data_dir)

    df = pd.DataFrame()
    df_idx = 0

    for sid in sessions:

        # Load session in place
        data_io.load_session(sid)
        cells_df = pd.read_csv(data_dir / f'{sid}_cells.csv', header=[0, 1], index_col=0)

        n_cells = cells_df.shape[0]
        print(f'{sid} - {n_cells} cells')
        for (prf, ec), rpf_df in data_io.burst_df.groupby(['repetition_frequency', 'electrode']):

            # Get train info
            train_info = rpf_df.iloc[0]
            train_id = train_info.train_id

            # Load data per cluster
            for cid in data_io.cluster_df.index.tolist():
                # Find distance to laser for this cluster in this trial
                laser_x, laser_y = train_info.laser_x, train_info.laser_y
                cluster_x, cluster_y = data_io.cluster_df.loc[cid, 'cluster_x'], data_io.cluster_df.loc[cid, 'cluster_y']
                df.at[df_idx, 'd'] = np.sqrt((cluster_x - laser_x) ** 2 + (cluster_y - laser_y) ** 2)
                df.at[df_idx, 'is_sig'] = cells_df.loc[cid, (train_id, 'is_significant')]
                df.at[df_idx, 'rpf'] = prf
                df_idx += 1

    df['prf_bin'] = pd.cut(df['rpf'],
                           bins=[1000, 1900, 2400, 3500, 4000, 4600, 5000, 6000, 7200],
                           right=False, include_lowest=True)
    df['prf_bin_number'] = df['prf_bin'].cat.codes


    for xbinwidth, xbinname in zip([25, 100], ['smallbin', 'largebin']):


        # Create figure
        fig = utils.simple_fig(
            width=0.7, height=1
        )

        for prf, prf_df in df.groupby('prf_bin'):
            if pd.isna(prf):
                continue

            prf_bin = np.mean([prf.left, prf.right])

            xbins = np.arange(0, 600, xbinwidth)
            x_plot = []
            y_plot = []

            for i in range(len(xbins)-1):
                x0 = xbins[i]
                x1 = xbins[i+1]

                y = prf_df.query(f'd >= {x0} and d < {x1}')['is_sig'].mean()

                x_plot.extend([x0, x1])
                y_plot.extend([y, y])

            # Plot data
            if prf_bin > 7500:
                prf_bin = 7500
            fig.add_scatter(
                x=x_plot, y=y_plot,
                mode='lines', line=dict(color=clrs.repetition_frequency(prf_bin)),
                showlegend=True,
                name=f'{prf_bin:.0f}',
            )

        fig.update_xaxes(
            title_text='d laser [um]',
            tickvals=np.arange(0, 600, 100),
            range=[0, 400],
        )
        fig.update_yaxes(
            title_text='% activated cells',
            tickvals=np.arange(0, 1.2, 0.2),
            range=[0, 1.05],
        )

        sname = figure_dir / 'pa_dc_min_max' / f'cell_count_vs_distance_{xbinname}'
        utils.save_fig(fig, sname, display=True)


def plot_single_cell_rasters(cell_overview):

    # setup parameters for plotting
    t_pre = 100  # ms
    t_post = 200  # ms

    # Connect to dataset
    data_io = DataIO(data_dir)

    # Loop through all the sessions
    for sid in sessions:

        # Load single session data
        data_io.load_session(sid)

        # Find cells to plot for this session
        co = cell_overview.query(f'session_id == "{sid}"')

        # Plot all individual cell / trial pairs
        for _, trial_to_plot in co.iterrows():

            # Extract codes for this cluster / trial pair
            cluster_id = trial_to_plot.cluster_id
            train_id = trial_to_plot.train_id
            electrode = trial_to_plot.electrode

            # Find all burst onsets for the dc min max series, on this electrode
            bursts_to_plot = data_io.burst_df.query(f'protocol == "pa_dc_min_max_series" '
                                                    f'and electrode == {electrode}')

            # Sort the data based on repetition frequency
            bursts_to_plot = bursts_to_plot.sort_values('repetition_frequency')

            # Create placeholders for data to plot
            x_plot, y_plot = [], []
            row_i = 0
            y_tck, y_txt = [], []

            fig = utils.make_figure(
                width=0.6, height=1,
                x_domains={1: [[0.1, 0.9]]},
                y_domains={1: [[0.1, 0.9]]},
            )

            for prf, bdf in bursts_to_plot.groupby('repetition_frequency'):

                # Find the recording name
                recording_name = bdf.iloc[0].recording_name

                # Get the spiketrain
                spiketrain = data_io.spiketimes[recording_name][cluster_id]

                # Add trial separator to data
                trial_sep_x = [-t_pre, t_post]
                trial_sep_y = [row_i, row_i]
                prf_clr = clrs.repetition_frequency(prf)

                # Add burst marker to plot data
                burst_marker_x = [0, 0, 10, 10]
                burst_marker_y = [row_i, row_i-bdf.shape[0], row_i-bdf.shape[0], row_i]

                y_tck.append(row_i - bdf.shape[0] / 2)
                y_txt.append(f'{prf/1000:.1f} kHz')

                fig.add_scatter(
                    x=trial_sep_x, y=trial_sep_y,
                    mode='lines',
                    line=dict(color=prf_clr, width=0.1, ),
                    showlegend=False,
                )

                fig.add_scatter(
                    x=burst_marker_x, y=burst_marker_y,
                    mode='lines', line=dict(color=prf_clr, width=0.1),
                    fill='toself',
                    fillcolor=clrs.repetition_frequency(prf, 0.8),
                    showlegend=False,
                )

                # append data per burst
                for _, burst_info in bdf.iterrows():
                    burst_onset = burst_info.burst_onset

                    # Find spikes in this burst
                    idx = np.where((spiketrain >= burst_onset - t_pre) & (spiketrain < burst_onset + t_post))[0]
                    spikes_this_burst = spiketrain[idx] - burst_onset

                    # Do a trick to plot the raster
                    spikes_this_burst = np.tile(spikes_this_burst, (3, 1))
                    spikes_this_burst[2, :] = np.nan

                    # Create the ydata to plot
                    row_to_plot = np.ones_like(spikes_this_burst) * row_i
                    row_to_plot[1, :] -= 1
                    row_to_plot[2, :] = np.nan

                    row_i -= 1

                    x_plot.append(spikes_this_burst.T.flatten())
                    y_plot.append(row_to_plot.T.flatten())

            x_plot = np.hstack(x_plot)
            y_plot = np.hstack(y_plot)


            fig.add_scatter(
                x=x_plot, y=y_plot,
                mode='lines', line=dict(color='black', width=0.5,),
                showlegend=False,
            )


            fig.update_xaxes(
                range=[-t_pre-1, t_post+1],
                tickvals=np.arange(-t_pre, t_post, 50),
                title_text='time [ms]'
            )
            fig.update_yaxes(
                range=[row_i, 0],
                tickvals=y_tck,
                ticktext=y_txt,
            )

            sname = figure_dir / 'pa_dc_min_max' / 'single_cell_rasters' / f'{cluster_id}_{electrode:.0f}'
            utils.save_fig(fig, sname, display=False)



def plot_single_channel_raw_traces():
    sid = '241108_A'
    data_io = DataIO(data_dir)
    data_io.load_session(sid)

    cid = 'uid_081124_022'

    burst_df = data_io.burst_df.query(f'protocol == "pa_dc_min_max_series" '
                                                    f'and electrode == 156')
    max_prf = burst_df.repetition_frequency.max()
    trial_info = burst_df.query(f'repetition_frequency == {max_prf}')
    recname = trial_info.iloc[0].recording_name

    from axorus.preprocessing.params import (data_sample_rate, data_type, data_nb_channels,
                                             data_trigger_channels, data_voltage_resolution,
                                             data_trigger_thresholds)
    recfile = r'C:\axorus\tmp2\241108_A_2_noblocker_pa.raw'
    data = np.memmap(recfile, dtype=data_type)
    n_samples = int(data.size / data_nb_channels)
    rec_duration = (n_samples / data_sample_rate) / 60  # [min]

    print(f'\treading data ({rec_duration:.0f} min)')

    # trigger_channel = data_trigger_channels['laser']

    # trigger_high = np.array([])

    # Define indices of current channel in data object
    # channel_index = np.arange(trigger_channel - 1, data.size, data_nb_channels)
    channel_index = np.arange(156, data.size, data_nb_channels)

    NPRE = 100  # time in ms
    NPOST = 200  # time in ms
    NSAMPLES = int(((NPRE + NPOST) / 1000) * data_sample_rate)
    n_samples_pre = (NPRE / 1000) * data_sample_rate

    # Load data per spike time
    all_idx = []
    for burst_i, bo in enumerate(trial_info.burst_onset.values):
        sref = (bo / 1000) * data_sample_rate
        i0 = int(sref - n_samples_pre)
        if i0 < 0:
            continue
        i1 = int(i0 + NSAMPLES)

        all_idx.append(np.arange(i0, i1, 1))


    all_idx = np.hstack(all_idx)
    chdata = data[channel_index[all_idx]].copy()
    chdata = chdata.reshape([trial_info.shape[0], NSAMPLES])
    chdata = chdata * data_voltage_resolution - 4096  # this is in uV now

    n_bursts = chdata.shape[0]
    x = (np.arange(0, chdata.shape[1], 1) / data_sample_rate) * 1000 - NPRE

    for i in range(n_bursts):

        ymin = chdata[i, :].min() - 1
        ymax = chdata[i, :].max() + 1

        spikes = data_io.spiketimes[recname][cid]
        brst_onset = burst_df.iloc[i].burst_onset
        idx = np.where((spikes >= brst_onset - NPRE) & (spikes < brst_onset + NPOST))[0]
        sp_plot = spikes[idx] - brst_onset

        fig = utils.make_figure(
            width=1, height=1,
            x_domains={1: [[0.1, 0.9]]},
            y_domains={1: [[0.1, 0.9]]},
        )

        fig.add_scatter(
            x=x, y=chdata[i, :].flatten(),
            mode='lines', line=dict(color='black', width=0.5,),
            showlegend=False,
        )

        fig.add_scatter(
            x=[0, 0, 10, 10], y=[ymin, ymax, ymax, ymin],
            mode='lines', line=dict(color='red', width=0.5),
            fill='toself',
            showlegend=False,
        )

        for s in sp_plot:
            fig.add_scatter(
                x=[s, s], y=[ymin, ymax],
                mode='lines', line=dict(color='blue', width=1),
                showlegend=False,
            )

        fig.update_yaxes(
            range=[ymin+1, ymax-1],
        )

        sname = figure_dir / 'pa_dc_min_max' / 'raw_test_data' / f'{i}'
        utils.save_fig(fig, sname, display=False)


if __name__ == '__main__':
    # cv = print_stats()
    # plot_cell_count_vs_distance_laser()
    # plot_cell_firing_rate_vs_distance_per_prf(cv)
    # plot_percent_modulated_cells()
    # plot_single_cell_rasters(cv)

    plot_single_channel_raw_traces()




