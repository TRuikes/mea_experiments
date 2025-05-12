from pathlib import Path
from axorus.data_io import DataIO


if __name__ == '__main__':
    data_dir = Path(r'C:\axorus\dataset')
    data_io = DataIO(data_dir)

    for sid in data_io.sessions:
        print(sid)
        data_io.load_session(sid, load_waveforms=False, load_pickle=True)
        data_io.dump_as_pickle()

