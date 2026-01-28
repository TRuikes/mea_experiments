import pandas as pd
import numpy as np
import utils
from scipy.signal import find_peaks


def read_oscilloscope_data(file):
    # Read oscilloscope metadata
    fs = pd.read_csv(file, usecols=[0, 1], header=None, names=['name', 'value'])
    fs.set_index('name', inplace=True)

    # Read oscilloscope data
    fd = pd.read_csv(file, usecols=[3, 4], header=None, names=['x', 'y'])
    return fs, fd


def plot_oscilloscope_data(fd, savename, data_specs=None):
    # Convert data to usable units
    t = fd.x.values * 1000  # ms
    v = fd.y.values  # a.u

    if data_specs is not None and 'title' in data_specs.keys():
        title = data_specs['title']
    else:
        title = ''

    # Visualize peak detection
    fig = utils.make_figure(
        height=1,
        width=1,
        x_domains={1: [[0.1, 0.9]]},
        y_domains={1: [[0.1, 0.9]]},
        subplot_titles={1: [title]},
        bg_color='black',
    )
    fig.add_scatter(
        x=t, y=v,
        mode='lines', line=dict(color='gold', width=1),
        name='raw voltage [a.u]'
    )

    if data_specs is not None:
        if 'detection_threshold' in data_specs.keys():
            thr = data_specs['detection_threshold']
            fig.add_scatter(
                x=t, y=np.ones_like(t) * thr,
                mode='lines', line=dict(color='red'),
                showlegend=False,
            )

        if 'peaks' in data_specs.keys():
            fig.add_scatter(
                x=t[data_specs['peaks']],
                y=v[data_specs['peaks']],
                mode='markers', marker=dict(color='green', size=5,
                                            line=dict(color='black', width=0.1)),
                showlegend=False,
            )

    if np.max(t) < 2:
        tv = np.arange(-10, 10, 0.2)
    else:
        tv = np.arange(-10, 10, 1)

    fig.update_xaxes(
        tickvals=tv,
        title_text='Time [ms]'
    )

    fig.update_yaxes(
        tickvals=np.arange(0, 100, 20),
        title_text='Voltage [V]',
    )

    fig.update_layout(
        legend_x=0.93,
        legend_y=0.3,
    )

    utils.save_fig(fig, savename, display=False)


def measure_repfreq(fd, threshold):
    ydata = fd['y']
    pks, pspcs = find_peaks(ydata, height=threshold, prominence=10,
                            distance=10)
    return pks
