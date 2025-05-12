from axorus.preprocessing.lib.filepaths import FilePaths
from axorus.preprocessing.lib.extract_triggers import extract_triggers
from axorus.preprocessing.lib.extract_phy_data import extract_phy_data
from axorus.preprocessing.lib.extract_trial_data import extract_trial_data
from axorus.preprocessing.dataset_sessions import dataset_sessions
from axorus.preprocessing.lib.create_dataset_object import create_dataset_object
from axorus.preprocessing.lib.extract_laser_position import detect_laser_position

for sid, s_specs in dataset_sessions.items():

    # Load the filepaths for the dataset
    filepaths = FilePaths(sid, laser_calib_week=s_specs['laser_calib_week'], local_raw_dir=s_specs['local_dir'])

    # Verify all files are there for this session
    filepaths.check_data()

    # Extract stimulation triggers
    extract_triggers(filepaths, update=False, visualize_detection=True)

    # # Extract trial data
    extract_trial_data(filepaths)

    # Extract manually sorted data
    if filepaths.has_manual_sorted_data:
        extract_phy_data(filepaths, update=True)

    create_dataset_object(filepaths, include_waveforms=True)
    # create_dataset_object(filepaths, include_waveforms=True)

    # Detect laser position
    # detect_laser_position(filepaths)   # doesnt work yet


