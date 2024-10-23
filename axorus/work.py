#%%

from axorus.data_io import DataIO
from pathlib import Path
import utils
import numpy as np

figdir = Path(r'F:\thijs\figures')
datadir = Path(r'F:\thijs\series_3\dataset')

data_io = DataIO(datadir)
sid = r'161024_A'
data_io.load_session(sid)

#%%

def plot_raster(dio: DataIO, cluster_id: str, ev_times: np.ndarray,
                rec_id: str):

    fig = utils.simple_fig(
        width=1, height=1,
    )

    stimes = dio.spiketimes[rec_id][cluster_id]
    x_plot, y_plot = [], []
    for ev_i, et in enumerate(ev_times):
        t0 = et - 200
        t1 = et + 400
        sidx = np.where((stimes >= t0) & (stimes <= t1))[0]

        for si in sidx:
            x_plot.extend([stimes[si] - et, stimes[si] - et, None])
            y_plot.extend([ev_i, ev_i+1, None])

    fig.add_scatter(
        x=x_plot, y=y_plot,
        mode='lines', line=dict(color='black', width=1)
    )

    fig.update_xaxes(
        tickvals=np.arange(-200, 500, 100),
        title_text=f'time (ms)'
    )

    return fig

#%%

cid = data_io.cluster_df.index.values[0]

for tid in data_io.train_df.train_id.unique():
    bursts = data_io.train_df.query('train_id == @tid').burst_onset.values

    for cid in data_io.cluster_df.index.values:
        f = plot_raster(data_io, cid, bursts, data_io.recording_ids[0])
        sname = figdir / cid / f'{tid}'
        utils.save_fig(f, sname, display=False)


