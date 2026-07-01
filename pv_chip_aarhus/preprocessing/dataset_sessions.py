"""
define the sessions for preprocessing here
laser_calib_week: which laser calibration file to use
fiber_connection: ?
local_dir: path to local copy of raw data, for faster reading (can set to None)


"""


dataset_sessions = {
    'test_data': dict(raw_data_dir='THISSHOULDBEADATAPATH'),  # add the parent dir containing the rawfiles whereon you ran the clustering
        # algorithm. e.g.: D:/ACJ/20260604/20260604 PV chip d-grease rd1 758 eye1 2nd half/Analysis/raw_filtered/
}