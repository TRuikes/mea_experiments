"""
define the sessions for preprocessing here
laser_calib_week: which laser calibration file to use
fiber_connection: ?
local_dir: path to local copy of raw data, for faster reading (can set to None)


"""


dataset_sessions = {
    # '241015_A': dict(laser_calib_week='week_41'),
    # '241016_A': dict(laser_calib_week='week_41', fiber_connection='CB1_14_C6'),
    # '241024_A': dict(laser_calib_week='week_41', fiber_connection='CB1_14_C6', local_dir=None),
    # '241108_A': dict(laser_calib_week='week_45', fiber_connection='', local_dir=r'E:\Axorus\tmp'),
    # '241211_A': dict(laser_calib_week='week_49', fiber_connection='', local_dir=r'E:\Axorus\tmp'),
    # '241213_A': dict(laser_calib_week='week_49', fiber_connection='', local_dir=r'C:\axorus\tmp2'),
    # '250403_A': dict(laser_calib_week='week_49', fiber_connection='', local_dir=r'E:\Axorus\tmp'),
    # '250520_A': dict(laser_calib_week='week_49', fiber_connection='', local_dir=r'C:\axorus\250520_A')
    # '250527_A': dict(laser_calib_week='week_49', fiber_connection='', local_dir=r'C:\axorus\250527_A\raw'),
    '250606_A': dict(laser_calib_week='week_49', fiber_connection='', local_dir=r'E:\Axorus\250606_A\raw')
}