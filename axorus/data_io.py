from pathlib import Path
import h5py
import pandas as pd
import threading


class DataIO:
    sessions = []
    recording_ids = []
    burst_df = pd.DataFrame()
    cluster_df = pd.DataFrame()
    spiketimes = {}
    waveforms = {}
    session_id = None

    def __init__(self, datadir: Path):
        self.datadir = datadir

        self.detect_sessions()

        self.lock = threading.Lock()
        self.is_locked = False

    def detect_sessions(self):
        self.sessions = [f.name.split('.')[0] for f in self.datadir.iterdir() if 'h5' in f.name]

    def load_session(self, session_id: str):
        assert session_id in self.sessions

        self.session_id = session_id

        burst_df = pd.DataFrame()
        cluster_df = pd.DataFrame()
        spiketimes = {}
        waveforms = {}
        rec_ids = []

        with h5py.File(self.datadir / f'{session_id}.h5', 'r') as f:

            for rec_id in f.keys():
                rec_ids.append(rec_id)
                for burst_id in f[rec_id]['laser'].keys():
                    new_id = f'{rec_id}-{burst_id}'

                    burst_df.at[new_id, 'rec_id'] = rec_id
                    burst_df.at[new_id, 'burst_id'] = burst_id

                    for k, v in f[rec_id]['laser'][burst_id].items():
                        if k in ['train_id']:
                            v_out = str(v[()]).split("'")[1]
                        else:
                            v_out = v[()]

                        burst_df.at[new_id, k] = v_out

                spiketimes[rec_id] = {}
                waveforms[rec_id] = {}
                for cluster_id in f[rec_id]['clusters'].keys():

                    for k, v in f[rec_id]['clusters'][cluster_id].items():
                        if k == 'spiketimes':
                            spiketimes[rec_id][cluster_id] = v[()]

                        elif k == 'waveforms':
                            waveforms[rec_id][cluster_id] = v[()]

                        else:
                            cluster_df.at[cluster_id, k] = v[()]

        for i, r in burst_df.iterrows():
            rn = r.recording_name
            burst_df.at[i, 'recording_name'] = str(rn).split("'")[1]

        self.burst_df = burst_df
        self.cluster_df = cluster_df
        self.spiketimes = spiketimes
        self.waveforms = waveforms
        self.recording_ids = rec_ids

    def lock_modification(self):
        self.lock.acquire()
        self.is_locked = True

    def unlock_modification(self):
        self.lock.release()
        self.is_locked = False


if __name__ == '__main__':
    import numpy as np
    datadir = Path(r'F:\thijs\series_3\dataset')
    data_io = DataIO(datadir)
    data_io.load_session(data_io.sessions[-1])

    print(data_io.spiketimes[data_io.recording_ids[0]].keys())
    print(data_io.cluster_df)
    # print(data_io.train_df.burst_offset - data_io.train_df.burst_onset)


