import pandas as pd
from pathlib import Path
from axorus.data_io import DataIO
import utils
import numpy as np
from axorus.preprocessing.project_colors import ProjectColors

session_id = '241024_A'
data_dir = Path(r'D:\Axorus\ex_vivo_series_3\dataset')
figure_dir = Path(r'C:\Axorus\figures')
data_io = DataIO(data_dir)
loadname = data_dir / f'{session_id}_cells.csv'

data_io.load_session(session_id)

cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)

#%%

names_to_copy = ['protocol', 'duty_cycle', 'burst_duration',
                 'laser_level', 'electrode', 'repetition_frequency',
                 'laser_power', 'laser_x', 'laser_y']
train_df = pd.DataFrame()
for tid in data_io.burst_df.train_id.unique():
    for n in names_to_copy:
        train_df.at[tid, n] = data_io.burst_df.query('train_id == @tid').iloc[0][n]

#%% Plot FR vs Pr per laser_level, per electrode stimulated

df_to_plot = train_df.query('protocol == "pa_dc_min_max_series"')
electrodes = df_to_plot.electrode.unique()
clrs = ProjectColors()

for electrode in electrodes:
    df = df_to_plot.query(f'electrode == {electrode}')

    fig = utils.make_figure(
        width=1,
        height=1,
        x_domains={1: [[0.1, 0.9]]},
        y_domains={1: [[0.1, 0.9]]}
    )

    for laser_level in df.laser_level.unique():
        dfl = df.query(f'laser_level == {laser_level}')

        Pr, firing_rate, firing_rate_se = [], [], []

        dfl = dfl.sort_values('duty_cycle')

        PRINTED_LEGEND = False

        for dc in dfl.duty_cycle.unique():
            tid = dfl.loc[dfl.duty_cycle == dc].iloc[0].name
            p = dfl.loc[dfl.duty_cycle == dc].iloc[0].repetition_frequency

            is_sig = cells_df[(tid, 'is_significant')]
            fr = cells_df[(tid, 'response_firing_rate')].values
            rtype = cells_df[(tid, 'response_type')]

            idx = np.where(pd.notna(fr) & (rtype == 'excited'))[0]
            fr_mean = np.mean(fr[idx])
            se = np.std(fr[idx]) / np.sqrt(idx.size)

            Pr.append(p)
            firing_rate.append(fr_mean)
            firing_rate_se.append(se)

        clr = clrs.laser_level(laser_level, 1)
        fig.add_scatter(
            x=Pr, y=firing_rate,
            error_y={
                'array': firing_rate_se,
            },
            mode='lines+markers',
            line=dict(color=clr, width=2),
            marker=dict(size=5, color=clr),
            showlegend=not PRINTED_LEGEND,
            name=f'laser level: {laser_level:.0f}',
        )

        PRINTED_LEGEND = True

    fig.update_xaxes(
        title_text='Pr [kHz]',
        tickvals=np.arange(0, 2e5, 2e3),
        ticktext=[f'{t:.1f}' for t in np.arange(0, 2e5, 2e3)/1e3]
    )

    fig.update_yaxes(
        tickvals=np.arange(0, 200, 20),
        title_text='FR [Hz]',
    )

    sname = figure_dir / session_id / 'pa_dc_min_max_series' / f'{electrode:.0f}_all'
    utils.save_fig(fig, sname)

#%%

def generate_circle_points(xin, yin, radius, num_points):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = radius * np.cos(theta) + xin
    y = radius * np.sin(theta) + yin
    return x, y

#%% Plot activation map per electrode

laser_level = 85
duty_cycle = 50

df_to_plot = train_df.query(f'laser_level == {laser_level} '
                            f'and duty_cycle == {duty_cycle}')

fig = utils.make_figure(
    width=1,
    height=1,
    x_domains={1: [[0.1, 0.9]]},
    y_domains={1: [[0.1, 0.9]]},
    equal_width_height='y',
)

ei = 0
for i, r in df_to_plot.iterrows():
    is_sig = cells_df[(i, 'is_significant')]
    idx = np.where(is_sig)
    cids = cells_df.index.values[idx]

    activated_pos = {}
    for cid in cids:
        c_x = data_io.cluster_df.loc[cid, 'cluster_x']
        c_y = data_io.cluster_df.loc[cid, 'cluster_y']

        if (c_x, c_y) not in activated_pos.keys():
            activated_pos[(c_x, c_y)] = 1
        else:
            activated_pos[(c_x, c_y)] += 1

    for k, v in activated_pos.items():
        fig.add_scatter(
            x=[k[0]], y=[k[1]],
            mode='markers',
            marker=dict(color=clrs.random_color(ei),
                        size=v*10),
            showlegend=False,
        )

    xplot, yplot = generate_circle_points(r.laser_x, r.laser_y, 100, 1000)
    print(r.electrode, r.laser_x, r.laser_y)
    fig.add_scatter(
        x=xplot, y=yplot,
        mode='lines',
        line=dict(color=clrs.random_color(ei),
                  dash='2px'),
        showlegend=False,
    )
        
    ei += 1

sname = figure_dir / session_id / 'activation_map'
utils.save_fig(fig, sname)





