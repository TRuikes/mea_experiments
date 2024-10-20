dataset_dir = r'E:\Photoacoustic\series_3'
data_trigger_channels = dict(
    # dmd=None,
    laser=128,
)
data_trigger_thresholds = dict(
    laser=1000,
)

data_sample_rate = 2e4
data_type = 'uint16'
data_nb_channels = 256
data_voltage_resolution = (2*4096) / (2**16)
nb_bytes_by_datapoint = 2
