import numpy as np
import os
from scipy.signal import butter, filtfilt


N_SAMPLES_PER_FRAME = 2000
MCD_GAPSIZE = 4064
N_CHANNELS = 256
SAMPLE_RATE = 20000
MCD_DATAOFFSET = 291904
MCD_DIGI_OFFSET = 287872
st = 291928
MCD_FRAMESIZE = 256 * N_SAMPLES_PER_FRAME * 2
MCD_DIGI_FRAMESIZE = 8 * N_SAMPLES_PER_FRAME * 2


def highpass_filter(data, sample_rate, cutoff=300, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=-1)  # assumes time is last dimension


def get_digital_data_indices(file_path, d_idx=0, i0=None, i1=None, t0=None, t1=None):
    frame_stride = MCD_FRAMESIZE + MCD_GAPSIZE
    n_frames = int(count_frames(file_path))

    idx = [
        np.arange(MCD_DIGI_OFFSET + (frame_stride * i),
                  MCD_DIGI_OFFSET + (frame_stride * i) + MCD_DIGI_FRAMESIZE,
                  1) + d_idx * 2
        for i in range(n_frames + 1)
    ]
    idx = np.hstack(idx)
    filesize = os.path.getsize(file_path)
    n_samples = (filesize / 2 - MCD_DIGI_OFFSET) / 256
    idx = idx[:int(n_samples)]

    if i0 is None and t0 is not None:
        sample_time = np.arange(0, idx.size) / SAMPLE_RATE
        to_keep = np.where((sample_time >= t0) & (sample_time <= t1))
    elif t0 is None and i0 is not None:
        sample_idx = np.arange(0, idx.size)
        to_keep = np.where((sample_idx >= i0) & (sample_idx <= i1))
    elif i0 is None and t0 is None:
        to_keep = np.arange(0, idx.size)
    else:
        raise ValueError('')

    return idx[to_keep]


def load_digital_data_from_mcd(file_path, d_idx=0, i0=None, i1=None,
                               t0=None, t1=None):
    byte_indices = get_digital_data_indices(file_path, d_idx=d_idx, i0=i0, i1=i1, t0=t0, t1=t1)

    data = read_channel_by_index(file_path, byte_indices)
    print(data)
    return data

def get_channel_indices(file_path, channel_index, filetype,
                        i0=None, i1=None, t0=None, t1=None):
    sample_rate = 20000
    frame_size = 256 * N_SAMPLES_PER_FRAME * 2
    n_frames = int(count_frames(file_path))

    if filetype == '.raw':
        data = np.memmap(file_path, dtype='uint16')
        idx = np.arange(2*channel_index, data.size, N_CHANNELS)
        n_samples = data.size / 256

    elif filetype == '.mcd':
        first_frame_offset = 291928  # mcd files start at this offset (following a header)
        frame_stride = frame_size + MCD_GAPSIZE
        filesize = os.path.getsize(file_path)
        n_samples =( filesize / 2 - first_frame_offset) / 256
        idx = [
            np.arange(first_frame_offset + (frame_stride * i),
                      first_frame_offset + (frame_stride * i) + frame_size,
                      N_CHANNELS)
            + 2 * channel_index
            for i in range(n_frames + 1)
        ]

        idx = np.hstack(idx)
        idx = idx[:int(n_samples) - 500]
        # idx = idx[idx < filesize]

    else:
        raise ValueError('')

    if i0 is None and t0 is not None:
        sample_time = np.arange(0, idx.size) / sample_rate
        to_keep = np.where((sample_time >= t0) & (sample_time <= t1))
    elif t0 is None and i0 is not None:
        sample_idx = np.arange(0, idx.size)
        to_keep = np.where((sample_idx >= i0) & (sample_idx <= i1))
    elif i0 is None and t0 is None:
        to_keep = np.arange(0, idx.size)
    else:
        raise ValueError('')

    return idx[to_keep]



def read_channel_by_index(filepath, byte_indices):
    """
    Reads uint16 values from specific byte indices using memory-mapped file access.
    """
    filesize = os.path.getsize(filepath)
    mm = np.memmap(filepath, dtype='uint8', mode='r', shape=(filesize,))

    # Read two bytes per index and interpret as little-endian uint16
    low = mm[byte_indices]
    high = mm[byte_indices + 1]
    return (high.astype(np.uint16) << 8) | low.astype(np.uint16)


def count_frames(filepath):
    """
    Returns the number of complete frames in the MCD file starting from a known offset.
    """

    # UTF-16LE encoding of "elec0"

    filesize = os.path.getsize(filepath)
    usable_bytes = filesize - st
    total_frame_size = MCD_FRAMESIZE + MCD_GAPSIZE
    return usable_bytes // total_frame_size


def recording_duration(filepath):
    return count_frames(filepath) * N_SAMPLES_PER_FRAME / SAMPLE_RATE


def load_channel_from_mcd_or_raw(file_path, channel_index, t0=None, t1=None,
                                 i0=None, i1=None, highpass_frequency=None):


    # Build byte indices for channel
    if '.mcd' in file_path:
        print('loading mcd')
        byte_indices = get_channel_indices(file_path, channel_index,
                                           '.mcd', t0=t0, t1=t1, i0=i0, i1=i1)
    elif '.raw' in file_path:
        print('loading raw')
        byte_indices = get_channel_indices(file_path, channel_index,
                                           '.raw', t0=t0, t1=t1, i0=i0, i1=i1)
    else:
        raise ValueError('unkown file_path extension')

    # Read data
    if highpass_frequency is not None:
        # filter data first
        all_indices = get_channel_indices(file_path, channel_index,
                                           '.mcd' if '.mcd' in file_path else 'raw',)
        data = read_channel_by_index(file_path, all_indices)
        data = data.astype(float)
        data_voltage_resolution = (2 * 4096) / (2 ** 16)
        data = data - np.iinfo('uint16').min + np.iinfo('int16').min
        data = data * data_voltage_resolution
        data[data == -4096] = 0

        data = highpass_filter(data, SAMPLE_RATE, highpass_frequency)

        if i0 is not None and t0 is None:
            data = data[i0:i1]
        elif i0 is None and t0 is not None:
            sample_time = np.arange(0, data.size) / SAMPLE_RATE
            to_keep = np.where((sample_time >= t0) & (sample_time < t1))
            data = data[to_keep]

    else:
        data = read_channel_by_index(file_path, byte_indices)

        # Convert data to voltage
        data_voltage_resolution = (2 * 4096) / (2 ** 16)
        data = data.astype(float)
        data = data - np.iinfo('uint16').min + np.iinfo('int16').min
        data = data * data_voltage_resolution
    return data

def find_data_offset(file):
    # Read and search the file for the marker
    with open(file, "rb") as f:
        data = f.read()

    # marker = b'elec0001'
    marker = b'digi0001'
    positions = []
    start = 0
    while True:
        index = data.find(marker, start)
        if index == -1:
            break
        first_byte_after = index + len(marker)
        positions.append(first_byte_after)
        start = index + 1

        if len(positions) > 5:
            break

    print(positions)
    pd = positions

    marker = b'elec0001'
    # marker = b'digi0001'
    positions = []
    start = 0
    while True:
        index = data.find(marker, start)
        if index == -1:
            break
        first_byte_after = index + len(marker)
        positions.append(first_byte_after)
        start = index + 1

        if len(positions) > 5:
            break


    print(positions)
    pe = positions

    # print(np.array(pe) - np.array(pd))
    # print(np.array(pd) - np.roll(np.array(pe), -1))
    # print(pe[1] - pd[1])
    # print((pd[2] - pe[1]) )#/ (N_SAMPLES_PER_FRAME * 2 * N_CHANNELS))
    # print((pe[2] - pd[1]) ) #/ (2 * N_SAMPLES_PER_FRAME))
    #
    # print((pe[2] - pd[1]) -(pd[2] - pe[1]))
    #


    # Order:
    # digid1; elecd1; digid 2; elecd2 ...
    frame_size_e = pe[2] - pe[1]
    frame_size_d = pd[2] - pd[1]

    print(f'f size e: {frame_size_e}, f size d: {frame_size_d}')

    block_size_e = pd[2] - pe[1] - 32
    block_size_d = pe[2] - pd[2] - 32

    print(f'block size e: {block_size_e}, block size d: {block_size_d}')
    print(f'diff block sizes: {(block_size_e - block_size_d) / (N_SAMPLES_PER_FRAME * 2)}')

    n_int = (block_size_e) / (N_SAMPLES_PER_FRAME * 2)
    n_int_d = (block_size_d) / (N_SAMPLES_PER_FRAME * 2)
    print(f'n int e: {n_int}, n int d: {n_int_d}')

    interval_d = pd[2] - pd[1]
    interval_e = pe[2] - pe[1]
    print(f'interval e {interval_e}, interval d {interval_d}')
    print(MCD_FRAMESIZE)

def main():
    mcd_path = r'C:\test_read_mcd\20250513_davis_OD1_2000ms_2-30_noblocker.mcd'
    # raw_path = r'C:\test_read_mcd\20250513_davis_OD1_2000ms_2-30_noblocker.raw'
    #
    # channel_index = 15
    # mcd_data = load_channel_from_mcd_or_raw(mcd_path, channel_index, 0, 20)
    # raw_data = load_channel_from_mcd_or_raw(raw_path, channel_index, 0, 20)
    #
    # data = np.memmap(raw_path, dtype='uint16')
    #
    # channel_index = np.arange(channel_index - 1, data.size, 256)
    #
    # rd2 = data[channel_index]
    #
    # # Compare
    # print(f"MCD samples: {len(mcd_data)}")
    # print(f"RAW samples: {len(raw_data)}")
    # print("Match:", np.array_equal(mcd_data, raw_data))

    find_data_offset(mcd_path)


if __name__ == "__main__":
    main()
