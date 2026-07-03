import os
###################################################################
#####################  Experiment Parameters  #####################
###################################################################
# name of your experiment for saving the triggers
exp = r'20260528_PV_chip_rd1_764_eye1_1st_half'

MEA = 1

# Link to the folder where spiking circus will look for the symbolic links "recording_0i.raw"
symbolic_link_directory = r"D:\ACJ\20260528\20260528 PV chip rd1 764 eye1 1st half\RAW"

# link to .GUI directory where phy extracts all arrays and data on spikes
phy_directory = r"D:\ACJ\20260528\20260528 PV chip rd1 764 eye1 1st half\RAW\rec_01_1-pow_N_P.GUI"

# Link to the actual raw files from the recording listed in the input_file
recording_directory = r"D:\ACJ\20260528\20260528 PV chip rd1 764 eye1 1st half\RAW"

# Link to the directory where output data should be saved
output_directory = r"D:\ACJ\20260528\20260528 PV chip rd1 764 eye1 1st half\Analysis"

# Link to the folder in which triggers will be saved. If doesn't exist, will be created.
triggers_directory = os.path.join(output_directory, "trigs")
raw_filtered_directory = os.path.join(output_directory, "raw_filtered")

# Ordered list of recording names with file extension
recording_names = [
    r"rec_01_1-pow_N_P.raw",
    r"rec_02_1-pow_N_P.raw",
    r"rec_03_2-pow_N_L.raw",
    r"rec_04_3-pow_LC_B.raw",
    r"rec_05_4-dur_N_P.raw",
    r"rec_06_5-dur_N_L.raw",
    r"rec_07_5-dur_N_L.raw",
    r"rec_08_6-dur_LC_B.raw",
    r"rec_09_7-freq_N_P.raw",
    r"rec_10_8-freq_N_L.raw",
    r"rec_11_9-freq_LC_B.raw",
    r"rec_12_10-del_N_B.raw",
    r"rec_13_11-flick_N_L.raw",
    r"rec_14_12-longstim_N_P.raw",
]

binary_source_path = 'binarysource1000Mbits'

registration_directory = r"D:\ACJ\20260528\20260528 PV chip rd1 764 eye1 1st half"
#Path to the checkerboard binary file used to generate stimuli
# binary_source_path = ''

# def find_files(path):
#     """
#     Function to get all recording files name from either a txt file name or a folder.

#     Input :
#         - path (string) : a .txt file path containing the recording .raw files name
#         - path (string) : a folder path containing all the recordings .raw files in alphabetic order
        
#     Output :
#         - (list) a list of strings of files names matching the recordings names
        
#     Possible mistakes :
#         - File names are written in .txt without the '.raw' extension
#         - Several files on the same line
#         - Wrong file/folder path
#         - Other files not in '.raw' extension in the folder
#         - Files names aren't ordered
#     """
#     if os.path.isfile(os.path.normpath(path)):                                      #Check if given path is a file and if it exist
#         with open(os.path.normpath(path)) as file:                                      #If yes, than open in variable "file"
#             return file.read().splitlines()                                                 #return the text of each line as a file name ordered from top to bottom
#     return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]       #If no, the path is considered as a folder and return the name of all the files in alphabetic order

# recording_names = find_files(recording_directory) #Do not use this unless you know why ! ! !

                                                                     ###################################################################
                                                                     ####################### Advanced Parameters #######################
                                                                     ###################################################################

"""
    Functions Parameters
    
Default values used in utils functions. If a function has a wrong behaviour, you may want to look in here.
"""

# Datatype used to open rawfiles recordings
dtype = 'uint16'

# Resolution of one step of mea signal amplitude in micro volts
voltage_resolution = 0.1042  # µV / DC level

# Size of a sample in bytes
nb_bytes_by_datapoint=2

# Number of time points to read in a trigger check
probe_size=1000000

# Maximal error admissible in sec for time gap between triggers 
maximal_jitter=0.25e-3

#Checkerboard sequences
nb_frames_by_sequence = 1200

#number of frames to look in for the lag
sta_temporal_dimension = 40

sta_smooth_value = 0.8

sta_treshold = 0.1

temporal_dimension = 30

"""
    SETUP parameters

Only change if you knwo what you are doing. Those parameters are following the setups specs of january 2023
"""

#the optimal threshhold for detecting stimuli onsets varies with the rig
if MEA==1: threshold  = 270e+3         
if MEA==2: threshold  = 50e+3          
if MEA==3: threshold  = 170e+3          
if MEA==4: threshold  = -3.14470e+5
#threshold = 200
poly_threshold = 350
laser_threshold = 100000

#256 for standard MEA, 17 for MEA1 Polychrome
nb_channels  = 256                

# number of triggers samples acquired per second
fs=20000

#MEA channel id containing polychrome triggers trace
poly_channel_id = 254

#MEA channel id containing laser triggers trace
laser_channel_id = 127

# --- Dead time control per modality ---
poly_time_before = 1     # ms
poly_time_after  = 10    # ms

laser_time_before = 1    # ms
laser_time_after  = 10   # ms

# --- Dead-time mode ---
# Options:
# 'onset'    → only onset-based dead time (current behaviour)
# 'interval' → full onset→offset interval masked
# 'hybrid'   → small windows around onset AND offset

dead_time_mode = 'hybrid'   # change per experiment or per run

# --- Offset handling ---
offset_time = 0          # keep existing

# --- Inclusion flags ---
include_poly  = True
include_laser = True    # change to True if desired

# --- Continuous stimulation handling ---
continuous_window = True     # activate LC / PC logic
continuous_time_before = 50  # ms around onset
continuous_time_after  = 50  # ms around offset

