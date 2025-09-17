import sys
from pathlib import Path
current_dir = Path().resolve()
sys.path.append(current_dir.as_posix().split('mea_experiments')[0] + 'mea_experiments')

from audrey.preprocessing.lib.filepaths import FilePaths
from audrey.preprocessing.lib.extract_triggers import extract_triggers
from audrey.preprocessing.lib.extract_phy_data import extract_phy_data
from audrey.preprocessing.lib.extract_trial_data import extract_trial_data
from audrey.preprocessing.dataset_sessions import dataset_sessions
from audrey.preprocessing.lib.create_dataset_object import create_dataset_object

for sid, s_specs in dataset_sessions.items():

    # Load the filepaths for the dataset
    filepaths = FilePaths(sid, laser_calib_week=s_specs['laser_calib_week'], local_raw_dir=s_specs['local_dir'])

    # # Verify all files are there for this session
    # filepaths.check_data()

    # # Extract stimulation triggers
    # extract_triggers(filepaths, update=False, visualize_detection=False,
    #                  recording_numbers_to_skip=s_specs['skip_triggers'])

    # # Extract trial data
    # extract_trial_data(filepaths)

    # # Extract manually sorted data
    # if filepaths.has_manual_sorted_data:
    #     extract_phy_data(filepaths, update=False)
    # else:
    #     print(f'NO SORTED DATA FOUND')

    create_dataset_object(filepaths, include_waveforms=False, 
                          recording_numbers_to_skip=s_specs['skip_triggers'])


    # Detect laser position
    # detect_laser_position(filepaths)   # doesnt work yet


