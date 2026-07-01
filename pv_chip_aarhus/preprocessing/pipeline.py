import sys
from pathlib import Path
current_dir = Path().resolve()
sys.path.append(current_dir.as_posix().split('mea_experiments')[0] + 'mea_experiments')

from pv_chip_aarhus.preprocessing.lib.filepaths import FilePaths
from pv_chip_aarhus.preprocessing.lib.extract_phy_data import extract_phy_data
from pv_chip_aarhus.preprocessing.lib.extract_trial_data import extract_trial_data
from pv_chip_aarhus.preprocessing.dataset_sessions import dataset_sessions
from pv_chip_aarhus.preprocessing.lib.create_dataset_object import create_dataset_object

for sid, s_specs in dataset_sessions.items():

    # Load the filepaths for the dataset
    filepaths = FilePaths(sid)

    # Extract trial data
    extract_trial_data(filepaths)
    # extract_phy_data(filepaths, update=True, waveform_extraction=False, raw_data_dir=s_specs['raw_data_dir'])
    create_dataset_object(filepaths, include_waveforms=False)

