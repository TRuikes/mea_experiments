from axorus.data_io import DataIO
from pathlib import Path

data_dir = Path(r'C:\thijs\sono_data\dataset')
data_io = DataIO(data_dir)

for sid in data_io.sessions:
    if 'Axorus' not in sid:
        continue
    print(sid)
    data_io.load_session(sid, load_waveforms=False, load_pickle=False)
    data_io.train_df['has_dmd'] = False
    data_io.train_df['has_laser'] = True

    names = ['burst_onset', 'burst_duration']
    for n  in names:
        data_io.burst_df[f'laser_{n}'] = data_io.burst_df[n]

    data_io.train_df['protocol_name'] = data_io.train_df['protocol']

    data_io.dump_as_pickle()