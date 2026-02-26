from utils import make_figure, save_fig
from sonogenetics.analysis.analysis_params import dataset_dir, figure_dir_analysis
import numpy as np

fig = make_figure(
    width=1, height=1,
    x_domains={1: [[0.1, 0.9]]},
    y_domains={1: [[0.1, 0.9]]},
)

x = np.arange(0, 10, 1)
fig.add_scatter(x=x, y=x)

fig.update_xaxes(
    tickvals=x,

)

sname = figure_dir_analysis / 'test'
save_fig(fig, sname, display=True)