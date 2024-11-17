from utils import load_obj, simple_fig, save_fig, save_obj, interp_color, make_figure
import numpy as np
import pandas as pd
from axorus.preprocessing.lib.get_probe_layout import get_probe_layout
from axorus.preprocessing.params import (nb_bytes_by_datapoint, data_nb_channels, data_sample_rate, dataset_dir,
                                         data_voltage_resolution, data_type, data_trigger_channels)
from axorus.preprocessing.lib.filepaths import FilePaths
import h5py


def read_trial_data_from_datasetfile(filepaths: FilePaths):
    burst_df = pd.DataFrame()

    with h5py.File(filepaths.dataset_file, 'r') as f:

        for rec_id in f.keys():

            if 'laser' in f[rec_id].keys():
                for burst_id in f[rec_id]['laser'].keys():
                    new_id = f'{rec_id}-{burst_id}'

                    burst_df.at[new_id, 'rec_id'] = rec_id
                    burst_df.at[new_id, 'burst_id'] = burst_id

                    for k, v in f[rec_id]['laser'][burst_id].items():
                        if v.dtype == 'int64':
                            v_out = int(v[()])
                        elif v.dtype == 'float64':
                            v_out = float(v[()])
                        else:
                            v_out = str(v[()]).split("'")[1]

                        burst_df.at[new_id, k] = v_out

    return burst_df


def generate_circle_points(xin, yin, radius, num_points):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = radius * np.cos(theta) + xin
    y = radius * np.sin(theta) + yin
    return x, y


scan_electrodes = [
    162, 227, 94, 24, 85, 14, 107,
    # 232, 164, 199, 155, 120, 42, 10,
    # 233, 137, 204, 125, 104, 33, 101,
]


def plot_laser_artefacts(filepaths: FilePaths, mea: str, laser_x, laser_y, channel_values,
                         recording_name, electrode):

    probe_layout = get_probe_layout(mea)

    x_plot = []
    y_plot = []
    z_plot = []

    fig = simple_fig(n_rows=1, n_cols=1, equal_width_height='y',
                           width=0.5, height=1)

    scan_electrodes = [
        162, 227, #94, 24, 85, 14, 107,
        # 232, 164, 199, 155, 120, 42, 10,
        # 233, 137, 204, 125, 104, 33, 101,
    ]

    for i, r in probe_layout.iterrows():

        x_plot.append(r['x'])
        y_plot.append(r['y'])

        if i in scan_electrodes:
            fig.add_annotation(x=x_plot[-1], y=y_plot[-1], showarrow=False,
                               text=f'{i}', font=dict(color='white', size=5))

        z_plot.append(channel_values[i-1])
        if np.isnan(channel_values[i-1]):
            z_plot.append(0)
        else:
            z_plot.append(channel_values[i-1])

    cmax = np.nanmax(z_plot)
    clrs = []
    for z in z_plot:
        if np.isnan(z):
            clrs.append('black')
        else:
            clrs.append(interp_color(cmax, z, ('sequential', 'Electric'), 1) )

    fig.add_scatter(
        x=x_plot, y=y_plot, mode='markers',
        marker=dict(color=clrs, size=8, line=dict(color='black', width=0.5)),
        showlegend=False,
    )

    xplot, yplot = generate_circle_points(laser_x, laser_y, 100, 1000)

    fig.add_scatter(
        x=xplot, y=yplot, mode='lines',
        marker=dict(color='rgba(255, 0,0, 1)', size=0.3),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        showlegend=False,
    )

    savename = filepaths.proc_pp_figure_output / 'artefact_position' / recording_name / f'{electrode:02f}'
    save_fig(fig, savename, display=True)


# noinspection PyTypeChecker
def plot_channel_data(filepaths, channel_data, mea, tinfo):
    from scipy.ndimage import median_filter

    channel_data_filtered = median_filter(channel_data, size=(15, 1))

    # Median filter along the 1st dimension with a window size of 3
    probe_layout = get_probe_layout(mea)

    x_positions = np.sort(probe_layout['x'].unique())
    y_positions = np.sort(probe_layout['y'].unique())
    n_columns = len(x_positions)
    n_rows = len(y_positions)

    x_offset = 0.05
    x_spacing = 0.01
    y_offset = 0.05
    y_spacing = 0.01
    ax_width = (1 - (n_columns-1) * x_spacing - 2 * x_offset) / n_columns
    ax_height = (1 - (n_rows - 1) * y_spacing - 2 * y_offset) / n_rows

    x_domains = {}
    y_domains = {}

    for row_i in range(n_rows):

        x_domains[row_i+1] = []
        y_domains[row_i+1] = []

        for col_i in range(n_columns):
            x0 = x_offset + (x_spacing + ax_width) * col_i
            y0 = 1 - y_offset - (y_spacing + ax_height) * row_i

            x_domains[row_i+1].append([x0, x0 + ax_width])
            y_domains[row_i+1].append([y0 - ax_height, y0])

    fig = make_figure(
        width=1, height=1,
        x_domains=x_domains, y_domains=y_domains
    )

    n_channels = channel_data.shape[0]

    # noinspection PyTypeChecker
    ymin = np.min(channel_data_filtered)
    ymax = np.max(channel_data_filtered)

    for i, r in probe_layout.iterrows():

        if i not in scan_electrodes and i != data_trigger_channels['laser']:
            continue

        if i == tinfo.electrode:
            clr = 'green'
        else:
            clr = 'black'

        ch_x = r.x
        ch_y = r.y

        col = np.where(x_positions == ch_x)[0][0] + 1
        row = np.where(y_positions == ch_y)[0][0] + 1

        if i != data_trigger_channels['laser']:
            y = channel_data_filtered[i-1, :]
        else:
            y = channel_data[i-1, :]

        x = np.arange(0, y.size, 1) / data_sample_rate

        fig.add_scatter(
            x=x, y=y,
            mode='lines', line=dict(color=clr, width=1),
            showlegend=False,
            row=row, col=col
        )

        if i != data_trigger_channels['laser']:
            fig.update_yaxes(
                range=[ymin, ymax],
                row=row, col=col,
            )

    tag = f'{tinfo.recording_name}_{tinfo.electrode}'
    savename = filepaths.proc_pp_figure_output / 'artefact_position' / 'raw_data' / tag
    save_fig(fig, savename, display=True)


def detect_laser_position(filepaths, plot=True):
    """

    """
    trial_overview = read_trial_data_from_datasetfile(filepaths)
    meas = trial_overview.mea.unique()
    mea = meas[0]
    if mea == '30/Aug':
        mea = '30_8'
    else:
        raise ValueError('errorito')

    assert len(meas) == 1
    # if filepaths.laser_artefact_position.is_file():
    #     print(f'\tlaser artefact all ready saved!')
    #     if plot:
    #         plot_laser_artefacts(filepaths, mea)
    #     return

    # Detect what mea was used
    probe_layout = get_probe_layout(mea)

    print(f'\tdetect laser position')

    # Load recording params
    n_sec_pre = 0.1
    n_sec_post = 0.3
    fs = data_sample_rate
    n_samples = int((n_sec_pre + n_sec_post) * fs)
    n_samples_pre = n_sec_pre * fs

    n_channels = data_nb_channels

    # Detect all laser positions in this recording
    # laser_overview = pd.read_csv(filepaths.proc_pp_triggers, index_col=0, header=0)
    stim_positions = trial_overview.electrode.unique()

    artefact_positions = pd.DataFrame()
    row_i = 0

    for (recording, electrode), df in trial_overview.groupby(['recording_name', 'electrode']):
        if electrode not in scan_electrodes:
            continue

        max_dc = df.duty_cycle.max()
        df_dc = df.query('duty_cycle == @max_dc')
        max_bd = df_dc.burst_duration.max()
        tinfo = df.query('duty_cycle == @max_dc and burst_duration == @max_bd').iloc[0]
        # t0 = tinfo.burst_onset

        if filepaths.local_raw_dir.exists():
            input_file = filepaths.local_raw_dir / f'{recording}.raw'
        else:
            input_file = filepaths.raw_raws / f'{recording}.raw'

        m = np.memmap(input_file.as_posix(), dtype=data_type)
        data_out = np.zeros((n_channels, n_samples), dtype=data_type)

        trigger_channel = data_trigger_channels['laser']

        for ch_i in range(n_channels):
            channel_index = np.arange(ch_i, m.size, data_nb_channels)
            i0 = int(tinfo.burst_onset / 1000 * fs - n_sec_pre * fs)
            i1 = int(tinfo.burst_onset / 1000 * fs + n_sec_post * fs)

            data_out[ch_i, :] = m[channel_index[i0:i1]]

        data_out = data_out.astype(float)
        data_out = data_out - np.iinfo('uint16').min + np.iinfo('int16').min
        data_out = data_out / data_voltage_resolution

        plot_channel_data(filepaths, data_out, mea, tinfo)

        channel_max = np.max(np.abs(data_out), axis=1)
        channel_max[255] = np.nan
        channel_max[126] = np.nan
        channel_max[125] = np.nan
        channel_max[254] = np.nan

        channel_max[data_trigger_channels['laser']] = np.nan
        channel_max[data_trigger_channels['dmd']] = np.nan

        channel_nrs = np.arange(0, channel_max.size)

        idx = ~np.isnan(channel_max)
        chm = channel_max[idx]
        chnr = channel_nrs[idx]

        idx = np.argsort(chm)[::-1]
        max_channels = chnr[idx[:9]]

        x = np.mean([probe_layout.loc[ch+1]['x'] for ch in max_channels])
        y = np.mean([probe_layout.loc[ch+1]['y'] for ch in max_channels])

        artefact_positions.at[row_i, 'recording'] = recording
        artefact_positions.at[row_i, 'stim_electrode'] = electrode
        artefact_positions.at[row_i, 'measured_laser_x'] = x
        artefact_positions.at[row_i, 'measured_laser_y'] = y

        row_i += 1

        plot_laser_artefacts(filepaths, mea, x, y, channel_max, recording, electrode)

    # if plot:
    #     plot_laser_artefacts(filepaths, mea)