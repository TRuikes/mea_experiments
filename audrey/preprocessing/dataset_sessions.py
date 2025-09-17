"""
define the sessions for preprocessing here
laser_calib_week: which laser calibration file to use
fiber_connection: ?
local_dir: path to local copy of raw data, for faster reading (can set to None)


"""


dataset_sessions = {
    '250904_A': dict(laser_calib_week='week_49', fiber_connection='', 
                     local_dir=None, skip_triggers=[1,2,3,4,10,11,12]), # Skip triggers are recording numbers
                                # to skip
}