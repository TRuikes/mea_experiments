import sys
from pathlib import Path
current_dir = Path().resolve()
print(current_dir)
sys.path.append(current_dir.as_posix().split('mea_experiments')[0] + 'mea_experiments')
print(current_dir.as_posix().split('mea_experiments')[0] + 'mea_experiments')
# from audrey.analysis.lib.data_io import DataIO
from mcd_lib import N_SAMPLES_PER_FRAME
from audrey.analysis.lib.data_io import DataIO