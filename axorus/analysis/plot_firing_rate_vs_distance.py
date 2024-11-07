#%%
import pandas as pd
from pathlib import Path
from axorus.data_io import DataIO
import utils
import numpy as np
sessions = ('161024_A',)

data_dir = Path(r'D:\thijs\series_3\dataset')
figure_dir = Path(r'C:\Axorus\figures')
data_io = DataIO(data_dir)
session_id = sessions[0]
loadname = data_dir / f'{session_id}_cells.csv'

data_io.load_session(session_id)

df = pd.read_csv(loadname, header=[0, 1], index_col=0)

#%%

duty_cycle = 90
laser_level = 85

for (laser_level, duty_cycle), group_df in data_io.train_df.groupby(['laser_level', 'duty_cycle']):
    electrodes = group_df.electrode.unique()

    y_offset = 0.1
    n_y = 3
    y_spacing = 0.15
    y_height = (1 - (2 * y_offset) - (n_y - 1) * y_spacing) / n_y
    y_domains = {}
    for i in range(n_y):
        y0 = 1 - y_offset - i * (y_height + y_spacing)
        y1 = y0 - y_height
        y_domains[i + 1] = [[y1, y0]]

    fig = utils.make_figure(
        width=0.5,
        height=1.2,
        x_domains={
            1: [[0.1, 0.9]],
            2: [[0.1, 0.9]],
            3: [[0.1, 0.9]],

        },
        y_domains=y_domains,
    )


    cluster_ids = data_io.cluster_df.index.values
    train_ids = data_io.train_df.train_id.unique()


    plot_row = 1
    for electrode, e_df in group_df.groupby('electrode'):
        train_id = e_df.iloc[0].train_id

        data_to_plot = pd.DataFrame()
        for cid in cluster_ids:
            data_to_plot.at[cid, 'response_firing_rate'] = df.loc[cid, (train_id, 'response_firing_rate')]
            laser_x = bursts.iloc[0].laser_x
            laser_y = bursts.iloc[0].laser_y

            cluster_x = data_io.cluster_df.loc[cid, 'cluster_x']
            cluster_y = data_io.cluster_df.loc[cid, 'cluster_y']

            d = np.sqrt((laser_x - cluster_x) ** 2 + (laser_y - cluster_y) ** 2)

            data_to_plot.at[cid, 'distance_to_laser'] = d

        pos = dict(row=plot_row, col=1)
        plot_row += 1
        fig.add_scatter(
            x=data_to_plot.distance_to_laser.values,
            y=data_to_plot.response_firing_rate.values,
            mode='markers', marker=dict(color='black', size=2),
            showlegend=False,
            **pos,
        )

        fig.update_xaxes(
            tickvals=np.arange(0, 1200, 200),
            title_text=f'distance to laser [um]' if pos['row'] == 2 else '',
            ticklen=1,
            **pos,
        )

        ymax = np.nanmax(data_to_plot.response_firing_rate.values)
        fig.update_yaxes(
            tickvals=np.arange(0, ymax, 30),
            title_text=f'Fr [Hz]',
            **pos,
        )

    sname = figure_dir / 'firing_rate_vs_distance' / f'{laser_level:.0f}_{duty_cycle:.0f}'
    utils.save_fig(fig, sname, display=False)

    #%%

