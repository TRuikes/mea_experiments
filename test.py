import numpy as np

def read_npy_file(file_path):
    """
    Reads a .npy file and returns the data.

    Parameters:
    - file_path (str): Path to the .npy file

    Returns:
    - data (ndarray): The data stored in the .npy file
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error reading .npy file: {e}")
        return None

def save_npy_file(file_path, data):
    # try:
    np.save(file_path, data, allow_pickle=True)
    # except:
    #     print(f'error saving .npy file')


def main():
    data = read_npy_file(r"C:\axorus\250520_A\spike_times.npy")



    idx = np.where(data != 0)[0]
    print('removing', data.size - idx.size)

    to_update = ['spike_times', 'spike_templates', 'amplitudes']
    data = data[idx]
    print(data.size)
    # print(data)
    savename = r"C:\axorus\250520_A\spike_times2.npy"
    save_npy_file( savename, data)

    data2 = read_npy_file(savename)
    idx = np.where(np.diff(data2.astype(float)) < 0)[0]
    print(idx)
    print('done')


if __name__ == '__main__':
    main()


