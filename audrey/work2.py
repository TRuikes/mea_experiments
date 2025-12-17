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


data_io = DataIO(data_dir)

for sid in data_io.sessions:
    data_io.load_session(sid)

    print(sid)
    print(data_io.burst_df.protocol.unique())