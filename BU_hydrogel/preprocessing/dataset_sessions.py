"""
define the sessions for preprocessing here
laser_calib_week: which laser calibration file to use
fiber_connection: ?
local_dir: path to local copy of raw data, for faster reading (can set to None)


"""


dataset_sessions = {
    # r'2025-12-16 rat P23H 3318 A': dict(laser_calib_week='week_49', fiber_connection='', 
    #                  local_dir=None, skip_triggers=[]), # Skip triggers are recording numbers
    #                             # to skip
    # r'2025-12-16 rat P23H 3318 B': dict(laser_calib_week='week_49', fiber_connection='', 
    #                  local_dir=None, skip_triggers=[]), 
    # r'2025-12-17 rat P23H 3153 A': dict(laser_calib_week='week_49', fiber_connection='',
    #                                     local_dir=None, skip_triggers=[])
    r'2026-01-22 rat LE 9999 A': dict(laser_calib_week='week_49', fiber_connection='',
                                    local_dir=None, skip_triggers=[])
}