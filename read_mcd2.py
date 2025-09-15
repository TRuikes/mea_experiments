import os

READDIGITAL = 0
DATASTART = 287864
# DATASTART = 291904
FRAMEBYTES = 1028064
FRAMEMARGINSTART = 32
ELECTRODEBYTES = 4000
FRAMEPOINTS = int(ELECTRODEBYTES / 2)
ELECTRODES_COUNT = 256

import numpy as np

def load_frames_numpy(filepath, first_frame, size):

    # Preallocate output array: (frames, points, electrodes)
    all_data = np.empty((size, FRAMEPOINTS, ELECTRODES_COUNT), dtype='<u2')

    with open(filepath, "rb") as file:
        print(f"Load Frames [{first_frame};+{size}]")

        # Seek to the start of the requested frames
        file.seek(DATASTART + first_frame * FRAMEBYTES)

        for frame in range(size):
            # Skip digital section
            file.seek(FRAMEMARGINSTART + ELECTRODEBYTES, 1)
            # Skip margin to analog
            file.seek(FRAMEMARGINSTART, 1)

            # Read all analog data for this frame at once
            num_values = FRAMEPOINTS * ELECTRODES_COUNT
            raw = file.read(num_values * 2)  # 2 bytes per value

            if len(raw) < num_values * 2:
                return all_data[:frame]  # premature EOF, return what we have

            # Interpret as unsigned int16 (little endian)
            data = np.frombuffer(raw, dtype='<u2')

            # Store into output array
            all_data[frame] = data.reshape(FRAMEPOINTS, ELECTRODES_COUNT)

    return all_data



def getFilesize(filepath):
    """
    Returns the size of the file at 'filepath' in bytes.
    If the file does not exist, returns None.
    """
    try:
        size = os.path.getsize(filepath)
        return size
    except OSError as e:
        print(f"Error: {e}")
        return None


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


N_SAMPLES_PER_FRAME = 2000
MCD_GAPSIZE = 4064
N_CHANNELS = 256
SAMPLE_RATE = 20000
MCD_DATAOFFSET = 291904
MCD_DIGI_OFFSET = 287872
st = 291928
MCD_FRAMESIZE = 256 * N_SAMPLES_PER_FRAME * 2
MCD_DIGI_FRAMESIZE = 8 * N_SAMPLES_PER_FRAME * 2

def count_frames(filepath):
    """
    Returns the number of complete frames in the MCD file starting from a known offset.
    """

    # UTF-16LE encoding of "elec0"

    filesize = os.path.getsize(filepath)
    usable_bytes = filesize - st
    total_frame_size = MCD_FRAMESIZE + MCD_GAPSIZE
    return usable_bytes // total_frame_size



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


def main():
    mcd_path = r'C:\test_read_mcd\20250513_davis_OD1_2000ms_2-30_noblocker.mcd'
    raw_path = r'C:\test_read_mcd\20250513_davis_OD1_2000ms_2-30_noblocker.raw'

    npts = 2000
    ec = 0
    size = getFilesize(mcd_path)
    data = size - DATASTART
    frameCount = data / FRAMEBYTES

    data_read = load_frames_numpy(mcd_path, 0, 2)
    print(data_read.shape, data_read[0:2, :10, ec:ec+2])
    # detect_format(filepath, DATASTART, frameCount)


    # try to read from raw file
    byte_indices = get_channel_indices(raw_path, ec, t0=0, t1=0.1, filetype='.raw')
    data_raw = read_channel_by_index(raw_path, byte_indices)

    print(data_raw[:10])
    # # Convert data to voltage
    # data_voltage_resolution = (2 * 4096) / (2 ** 16)
    # data_raw = data_raw.astype(float)
    # data_raw = data_raw - np.iinfo('uint16').min + np.iinfo('int16').min
    # data_raw = data_raw * data_voltage_resolution


if __name__ == '__main__':
    main()