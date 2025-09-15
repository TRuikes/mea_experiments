from mcd_lib import load_channel_from_mcd_or_raw, recording_duration, load_digital_data_from_mcd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import utils
from pathlib import Path
from tqdm import tqdm


def detect_threshold_crossings(signal, fs, threshold, dead_time):
    """
    Detect threshold crossings in a signal with dead time and return them as an array

    The signal transitions from a sample above the threshold to a sample below the threshold for a detection and
    the last detection has to be more than dead_time apart from the current one.

    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param threshold: The threshold for the signal
    :param dead_time: The dead time in seconds.
    """
    dead_time_idx = dead_time * fs
    threshold_crossings = np.diff((signal <= threshold).astype(int) > 0).nonzero()[0]
    distance_sufficient = np.insert(np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    while not np.all(distance_sufficient):
        # repeatedly remove all threshold crossings that violate the dead_time
        threshold_crossings = threshold_crossings[distance_sufficient]
        distance_sufficient = np.insert(np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    return threshold_crossings


def main():
    mcd_path = r'C:\test_read_mcd\20250513_davis_OD1_2000ms_2-30_noblocker.mcd'
    raw_path = r'C:\test_read_mcd\20250513_davis_OD1_2000ms_2-30_noblocker.raw'

    # mcd_path = r'C:\test_read_mcd\20250228_rat2631_OD1_500ms_10sec_poly410_5-30.mcd'
    # mcd_path = r'C:\test_read_mcd\20250513_davis_OD1_poly600_2000ms_5-100_noblocker.mcd'
    sample_rate = 20000
    channel_index = 200
    t0 = 0
    t1 = 5
    i0 = 0
    i1 = 10 * sample_rate


    # rec_dur = recording_duration(raw_path)
    # print(f'recording duration: {rec_dur:.2f}s ')

    rec_dur = recording_duration(mcd_path)
    print(f'recording duration: {rec_dur:.2f}s ')

    # # Detect spikes
    # signal = load_channel_from_mcd_or_raw(mcd_path, channel_index)

    # for d in range(100):
    digi_data = load_digital_data_from_mcd(
        mcd_path, d_idx=3, t0=t0, t1=t1
    )

    threshold_crossings = np.diff((digi_data > 0).astype(int) > 0).nonzero()[0]


    fig = go.Figure()
    fig.add_scatter(
        x=np.arange(0, digi_data.size) / sample_rate + t0,
        y=digi_data,
    )

    theshold = np.max(digi_data) / 2
    fig.add_scatter(
        x=threshold_crossings / sample_rate + t0,
        y=np.ones_like(threshold_crossings) * theshold,
        mode='markers', marker=dict(color='red', size=5)
    )
    # fig.show()
    savename = Path(rf'C:\test_read_mcd\figures\ch_{1}')
    utils.save_fig(fig, savename, display=True)
    # spiketimes = {}
    #
    # for channel_index in tqdm(range(256)):
    #
    #     # Get dat afor plotting
        mcd_data = load_channel_from_mcd_or_raw(mcd_path, channel_index) #,  i0=i0, i1=i1)  #t0=t0, t1=t1)
    #     mcd_data_filtered = load_channel_from_mcd_or_raw(
    #         mcd_path, channel_index, t0=t0, t1=t1, highpass_frequency=200)
    #
    #     # noise_std = np.std(mcd_data_filtered)
    #     noise_mad = np.median(np.absolute(mcd_data_filtered)) / 0.6745
    #     # print('Noise Estimate by Standard Deviation: {0:g} V'.format(noise_std))
    #     # print('Noise Estimate by MAD Estimator     : {0:g} V'.format(noise_mad))
    #
    #     spike_threshold = -3 * noise_mad  # roughly -30 µV
    #     # print(f'Spike threshold: {spike_threshold:.1f} V')
    #     # raw_data = load_channel_from_mcd_or_raw(raw_path, channel_index) # , i0=i0, i1=i1)  #t0=t0, t1=t1)
    #
    #     crossings = detect_threshold_crossings(mcd_data_filtered, sample_rate, spike_threshold, 0.003)  # dead time of 3 ms
    #     crossings_time = crossings / sample_rate
    #
    #     spiketimes[channel_index] = crossings_time
    #
    #     # raw_data = raw_data[:mcd_data.size]
    #     # print(mcd_data.shape, 'mc shape')
    #     # print(raw_data.shape, 'raw dshape')
    #
    #     # Optional: compare
    #     # match = np.array_equal(mcd_data, raw_data)
    #     # print("Frames match:", match)
    #     # print(np.where(raw_data != mcd_data)[0])
    #     # New code to plot the data
    #     # num_samples = len(mcd_data)
    #     # time = np.linspace(t0, t1, num=mcd_data_filtered.size, endpoint=False)
    #
    #     # fig = go.Figure()
    #
    #     # fig.add_trace(go.Scatter(
    #     #     x=time,
    #     #     y=mcd_data,
    #     #     mode='lines',
    #     #     name='MCD Data',
    #     #     line=dict(color='blue', width=0.5)
    #     # ))
    #
    #     # fig.add_trace(go.Scatter(
    #     #     x=time,
    #     #     y=mcd_data_filtered,
    #     #     mode='lines',
    #     #     name='MCD Data filtered',
    #     #     line=dict(color='red', width=0.5)
    #     # ))
    #     # fig.add_trace(go.Scatter(
    #     #     x=time,
    #     #     y=np.ones_like(time)*spike_threshold,
    #     #     mode='lines',
    #     #     name='MCD Data filtered',
    #     #     line=dict(color='red', width=0.5)
    #     # ))
    #     #
    #     # fig.add_scatter(
    #     #     x=crossings_time + t0, y=np.ones_like(crossings) * spike_threshold,
    #     #     mode='markers',
    #     #     marker=dict(color='black', size=2),
    #     # )
    #     #
    #     # fig.update_layout(
    #     #     title=f'Channel {channel_index} from {t0}s to {t1}s',
    #     #     xaxis_title='Time (s)',
    #     #     yaxis_title='Amplitude',
    #     #     # yaxis=dict(range=[-100, 100]),
    #     #     width=1000,
    #     #     height=400,
    #     #     legend=dict(x=0.01, y=0.99)
    #     # )
    #     #
    #     # # fig.show()
    #     # utils.save_fig(fig, Path(rf'C:\test_read_mcd\figures\ch_{channel_index}'),
    #     #                display=False)


if __name__ == "__main__":
    main()