"""
define the sessions for preprocessing here
laser_calib_week: which laser calibration file to use
fiber_connection: ?
local_dir: path to local copy of raw data, for faster reading (can set to None)


"""
# add the parent dir containing the rawfiles whereon you ran the clustering

dataset_sessions = {
    
    #'20260601_eye1_1st': dict(raw_data_dir=r'D:\ACJ\20260601\20260601 PV chip rd1 770 eye1 1st half\Analysis\raw_filtered'),
    #'20260603_eye2_1st': dict(raw_data_dir=r'D:\ACJ\20260603\20260603 Fake chip rd1 766 eye2 1st half\Analysis\raw_filtered'), # PROBLEM for DEL2 which was only laser instead of both
    '20260603_eye2_2nd': dict(raw_data_dir=r'D:\ACJ\20260603\20260603 Fake chip d-grease rd1 766 eye2 2nd half\Analysis\raw_filtered'),
    #'20260604_eye1_2nd': dict(raw_data_dir=r'D:\ACJ\20260604\20260604 PV chip d-grease rd1 758 eye1 2nd half\Analysis\raw_filtered'),  
    #'20260604_eye2_2nd': dict(raw_data_dir=r'D:\ACJ\20260604\20260604 Fake chip d-grease rd1 758 eye2 2nd half\Analysis\raw_filtered'),
    #'20260527_eye2_1st': dict(raw_data_dir=r'D:\ACJ\20260527\20260527 No chip rd1 763 eye2 1st half\Analysis\raw_filtered'),   
        # algorithm. e.g.: D:/ACJ/20260604/20260604 PV chip d-grease rd1 758 eye1 2nd half/Analysis/raw_filtered/
}