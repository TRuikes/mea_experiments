"""
define the sessions for preprocessing here
laser_calib_week: which laser calibration file to use
fiber_connection: ?
local_dir: path to local copy of raw data, for faster reading (can set to None)


"""


dataset_sessions = {
    # '241015_A': dict(laser_calib_week='week_41'),
    # '241016_A': dict(laser_calib_week='week_41', fiber_connection='CB1_14_C6'),
    # '241024_A': dict(laser_calib_week='week_41', fiber_connection='CB1_14_C6'),
    '241108_A': dict(laser_calib_week='week_45', fiber_connection='', local_dir=r'C:\axorus\tmp')
}