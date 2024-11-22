from axorus.preprocessing.lib.filepaths import FilePaths
from axorus.preprocessing.params import nb_bytes_by_datapoint, data_nb_channels, data_sample_rate, dataset_dir, data_voltage_resolution, data_type
from scipy.signal import butter, filtfilt
import pandas as pd
from pathlib import Path
from axorus.preprocessing.project_colors import ProjectColors
from matplotlib.pyplot import subplot
from scipy.ndimage import gaussian_filter

from axorus.data_io import DataIO
import utils
import numpy as np
from axorus.preprocessing.params import data_sample_rate, data_type, data_nb_channels

session_id = '241108_A'
data_dir = Path(r'F:\Axorus\ex_vivo_series_3\dataset')
figure_dir = Path(r'C:\Axorus\figures')
data_io = DataIO(data_dir)
loadname = data_dir / f'{session_id}_cells.csv'

data_io.load_session(session_id)

cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)

clrs = ProjectColors()

# Define filter parameters
fs = 1000  # Sampling frequency in Hz (adjust to your actual sampling frequency)
cutoff = 50  # Cutoff frequency in Hz
order = 4  # Filter order

filepaths = FilePaths('241108_A', local_raw_dir=r'E:\Axorus\tmp')

uid = 'uid_081124_001'
cluster_info = data_io.cluster_df.loc[uid]
cluster_channel = cluster_info.ch
bursts = data_io.burst_df.query('protocol == "pa_dc_min_max_series"')

channels_to_plot = [135, 120, 83]


for (tid, _), df in data_io.burst_df.groupby(['train_id', 'electrode']):
    # bursts = data_io.burst_df.query('protocol == "pa_temporal_series"')
    # rec_id = bursts.iloc[0].rec_id
    rec_id = df.iloc[0].rec_id

    # Retreive path to rawfile
    if filepaths.local_raw_dir is not None:
        input_file = filepaths.local_raw_dir / f'{rec_id}.raw'
    else:
        input_file = filepaths.raw_dir / f'{rec_id}.raw'

    # Open datafile
    m = np.memmap(input_file.as_posix(), dtype=data_type)

    # General setup of figure
    n_rows = 1
    n_cols = 1

    x_domains = {}
    y_domains = {}
    subplot_titles = {}
    x_offset = 0.05
    y_offset = 0.1
    x_spacing = 0.1
    y_spacing = 0.1

    x_width = (1 - (n_cols - 1) * x_spacing - 2 * x_offset) / n_cols
    y_height = (1 - (n_rows - 1) * y_spacing - 2 * y_offset) / n_rows

    for row_i in range(n_rows):
        x_domains[row_i + 1] = []
        y_domains[row_i + 1] = []
        subplot_titles[row_i + 1] = []

        for col_j in range(n_cols):
            x0 = x_offset + col_j * (x_spacing + x_width)
            x_domains[row_i + 1].append([x0, x0 + x_width])

            y1 = 1 - y_offset - row_i * (y_spacing + y_height)
            y_domains[row_i + 1].append([y1 - y_height, y1])

    fig = utils.make_figure(
        width=1, height=1.2,
        x_domains=x_domains,
        y_domains=y_domains,
    )

    for ei, ec in enumerate(channels_to_plot):
        cluster_channel = ec - 1
        channel_index = np.arange(cluster_channel, m.size, data_nb_channels,
                                  dtype=int)

        # Extract a burst
        burst_onset = bursts.iloc[2].burst_onset / 1000  # in [s]
        t_pre = 0.1  # [s]
        t_after = 0.3  # [s]
        n_pre = t_pre * data_sample_rate  # [samples]
        n_after = t_after * data_sample_rate  # [samples]
        n_samples = n_pre + n_after

        i_ref = burst_onset * data_sample_rate  # [samples]
        i0 = int(i_ref - n_pre)
        i1 = int(i_ref + n_after)

        burst_data = m[channel_index[i0:i1]]

        # Design the Butterworth highpass filter
        b, a = butter(order, cutoff, btype='high', fs=data_sample_rate)

        # Apply the filter to the data
        filtered_burst_data = filtfilt(b, a, burst_data)

        y_min = np.min(burst_data)
        y_max = np.max(burst_data)

        time = ((np.arange(i0, i1, 1) / data_sample_rate)) * 1000

        fig.add_scatter(
            x=time, y=burst_data + ei * 0.001 * (y_max),
            mode='lines', line=dict(width=1),
            showlegend=False,
        )
    #
    bo = burst_onset * 1000
    fig.add_scatter(
        x=[bo, bo, bo+10, bo+10], y=[y_min, y_max, y_max, y_min],
        fillcolor='rgba(255, 0, 0, 0.3)', fill='toself',
        mode='lines', line=dict(width=0),
        showlegend=False,
    )

    fig.update_xaxes(
        # range=[-100, 200],
        tickvals=[time[0], time[-1]]
    )

    fig.update_yaxes(
        range=[y_min, y_max],
    )

    sname = figure_dir / session_id / 'work' / 'test'
    utils.save_fig(fig, sname, display=True)

    break
