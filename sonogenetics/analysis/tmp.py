from sonogenetics.analysis.dev_mua_analysis import read_triggers, get_channel_index, get_filtered_daa_and_spikes

from sonogenetics.preprocessing.params import (data_sample_rate, data_type, data_nb_channels,
                                         data_trigger_channels, data_voltage_resolution,
                                         data_trigger_thresholds)
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

filename = r"E:\sono\2026-03-25 mouse c57 617 Mekano6 A\raw\rec_2_A_20260325_pa_intensity_test.raw"


# Load channel data
trigger_channel = 'laser'
data = np.memmap(filename, dtype=data_type)
filename = Path(filename)
n_samples = int(data.size / data_nb_channels)
rec_duration = (n_samples / data_sample_rate) / 60  # [min]


# Define indices of current channel in data object
triggers = read_triggers(data, trigger_channel)
n_trains = len(triggers['train_onsets'])
n_bursts = len(triggers['burst_onsets'])

# Print file information
print(f'\t\t\t{filename.name}\n')
print(f'\trecording duration: {rec_duration:.0f} min\n')
print(f'\tdetected {n_trains} trains on {trigger_channel} trigger')
print(f'\tdetected {n_bursts} bursts on {trigger_channel} trigger')


train_onsets = triggers['train_onsets']
burst_onsets = triggers['burst_onsets']
idx = np.where(
    (burst_onsets >= 758582.5625) &
    (burst_onsets < 787457.4375)
)[0]
print(idx.size)

trigger_type = 'laser'
d = []
for ti, t in tqdm(enumerate(burst_onsets[idx])):
    d.append(get_filtered_daa_and_spikes(data, 174, t / 1e3, ti, 0))


plt.figure(figsize=(10, 6))

for i, item in enumerate(d):
    segment = item['segment']

    # Optional: create a time axis if you want proper scaling
    x = np.arange(len(segment)) / 20

    plt.plot(x, segment, alpha=0.5)  # alpha helps with overlap

plt.xlabel("Sample Index")
plt.ylabel("Segment Value")
plt.title("All Segments Overlayed")
plt.tight_layout()
plt.show()