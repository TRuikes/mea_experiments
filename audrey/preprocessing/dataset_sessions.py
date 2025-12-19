"""
define the sessions for preprocessing here
laser_calib_week: which laser calibration file to use
fiber_connection: ?
local_dir: path to local copy of raw data, for faster reading (can set to None)
skip_triggers: list of recording numbers to skip. recording numbers are recognised as _{RECNR:03d}_

"""


dataset_sessions = {
    # '250904_A': dict(laser_calib_week='week_49', fiber_connection='',
    #                  local_dir=None, skip_triggers=[1,2,3,4,10,11,12]), # Skip triggers are recording numbers to skip
    # '251014_A': dict(laser_calib_week='week_49', fiber_connection='',
    #                 local_dir=None, skip_triggers=[1, 2, 3, 4]), # Skip triggers are recording numbers to skip),
    # '251014_B': dict(laser_calib_week='week_49', fiber_connection='',
    #                 local_dir=None, skip_triggers=[1, 2, 3, 4, 8]), # Skip triggers are recording numbers to skip),
    '251015': dict(laser_calib_week='week_49', fiber_connection='',
                    local_dir=None, skip_triggers=[1,2,3,4,8]), # Skip triggers are recording numbers to skip),
}