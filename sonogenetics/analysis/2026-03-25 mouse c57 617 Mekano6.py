import sys
from pathlib import Path
from utils import make_figure, save_fig

current_dir = Path().resolve()
sys.path.append(current_dir.parent.parent.as_posix())
import os
import pandas as pd
from sonogenetics.analysis.data_io import DataIO
import numpy as np
from sonogenetics.project_colors import ProjectColors
from sonogenetics.analysis.analysis_params import dataset_dir, figure_dir_analysis

# Load data
if not os.path.exists(figure_dir_analysis):
    os.makedirs(figure_dir_analysis)

data_io = DataIO(dataset_dir)
session_id = '2026-03-25 mouse c57 617 Mekano6 A'

figure_dir_analysis = figure_dir_analysis / session_id
print(session_id)
data_io.load_session(session_id, load_pickle=False, load_waveforms=False)
data_io.dump_as_pickle()

loadname = dataset_dir / f'{session_id}_cells.csv'
cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
clrs = ProjectColors()

# Print available recording ids
print("Available recording ids:")
for rec_id in data_io.recording_ids:
    print(f"- {rec_id}")


from sonogenetics.analysis.plot_responses_single_session import raster_per_protocol_master

for i, r in data_io.burst_df.iterrows():
    data_io.burst_df.at[i, 'protocol'] = r.recording_name

raster_per_protocol_master(data_io=data_io)