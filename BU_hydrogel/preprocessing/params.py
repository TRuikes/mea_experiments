dataset_dir = r'C:\thijs\bu_hydrogel'

data_trigger_channels = dict(
    dmd=127,
    laser=128,
)
data_trigger_thresholds = dict(
    laser=1000,
    dmd=1000,

)

data_sample_rate = 2e4
data_type = 'uint16'
data_nb_channels = 256
data_voltage_resolution = (2*4096) / (2**16)
nb_bytes_by_datapoint = 2

manuall_edited_sessions = ['2025-12-17 rat P23H 3153 A']