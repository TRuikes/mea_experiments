from axorus.preprocessing.lib.filepaths import FilePaths
from axorus.preprocessing.lib.extract_triggers import extract_triggers
from axorus.preprocessing.lib.extract_phy_data import extract_phy_data
from axorus.preprocessing.lib.extract_trial_data import extract_trial_data


sessions = (
    # '151024_A',
    '161024_A',
)


for sid in sessions:

    # Load the filepaths for the dataset
    filepaths = FilePaths(sid)

    # Verify all files are there for this session
    filepaths.check_data()

    # Extract stimulation triggers
    extract_triggers(filepaths, update=False, visualize_detection=False)

    # Extract trial data
    extract_trial_data(filepaths)

    if filepaths.has_manual_sorted_data:
        extract_phy_data(filepaths, update=False)



    # assert filepaths.has_manual_sorted_data

