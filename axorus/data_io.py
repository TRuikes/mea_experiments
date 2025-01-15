from pathlib import Path
import h5py
import pandas as pd
import threading
import pickle

class DataIO:
    sessions = []
    recording_ids = []
    burst_df = pd.DataFrame()
    cluster_df = pd.DataFrame()
    spiketimes = {}
    waveforms = {}
    session_id = None
    pickle_file = None

    def __init__(self, datadir: Path):
        self.datadir = datadir

        self.detect_sessions()

        self.lock = threading.Lock()
        self.is_locked = False

    def detect_sessions(self):
        self.sessions = [f.name.split('.')[0] for f in self.datadir.iterdir() if 'h5' in f.name]

    def load_session(self, session_id: str, load_waveforms=False,
                     load_pickle=True):
        assert session_id in self.sessions

        self.session_id = session_id
        self.pickle_file = self.datadir / f'{session_id}.pickle'

        if load_pickle and self.pickle_file.is_file():
            self.load_pickle()

        else:
            burst_df = pd.DataFrame()
            cluster_df = pd.DataFrame()
            spiketimes = {}
            waveforms = {}
            rec_ids = []

            if load_waveforms is False:
                readfile = self.datadir / f'{session_id}.h5'
            else:
                readfile = self.datadir / f'{session_id}_waveforms.h5'

            with h5py.File(readfile, 'r') as f:

                for rec_id in f.keys():
                    rec_ids.append(rec_id)

                    if 'laser' in f[rec_id].keys():
                        for burst_id in f[rec_id]['laser'].keys():
                            new_id = f'{rec_id}-{burst_id}'

                            burst_df.at[new_id, 'rec_id'] = rec_id
                            burst_df.at[new_id, 'burst_id'] = burst_id

                            for k, v in f[rec_id]['laser'][burst_id].items():
                                if v.dtype == 'int64':
                                    v_out = int(v[()])
                                elif v.dtype == 'float64':
                                    v_out = float(v[()])
                                else:
                                    v_out = str(v[()]).split("'")[1]

                                burst_df.at[new_id, k] = v_out

                    spiketimes[rec_id] = {}
                    waveforms[rec_id] = {}
                    for cluster_id in f[rec_id]['clusters'].keys():

                        for k, v in f[rec_id]['clusters'][cluster_id].items():
                            if k == 'spiketimes':
                                spiketimes[rec_id][cluster_id] = v[()]

                            elif k == 'waveforms':

                                if load_waveforms:
                                    waveforms[rec_id][cluster_id] = v[()]

                            else:
                                cluster_df.at[cluster_id, k] = v[()]

            # for i, r in burst_df.iterrows():
            #     rn = r.recording_name
            #     burst_df.at[i, 'recording_name'] = str(rn).split("'")[1]

            self.burst_df = burst_df
            self.cluster_df = cluster_df
            self.spiketimes = spiketimes
            if load_waveforms:
                self.waveforms = waveforms
            else:
                self.waveforms = None
            self.recording_ids = rec_ids

    def lock_modification(self):
        self.lock.acquire()
        self.is_locked = True

    def unlock_modification(self):
        self.lock.release()
        self.is_locked = False

    def dump_as_pickle(self):
        data_to_pickle = dict(
            spiketimes=self.spiketimes,
            cluster_df=self.cluster_df,
            burst_df=self.burst_df,
            recording_ids=self.recording_ids,
        )
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(data_to_pickle, f)

    def load_pickle(self):
        """Load data from a pickle file into the current instance."""
        with open(self.pickle_file, 'rb') as f:
            loaded_instance = pickle.load(f)

            for k, v in loaded_instance.items():
                self.__setattr__(k, v)

if __name__ == '__main__':
    data_dir = Path(r'C:\axorus\dataset')
    data_io = DataIO(data_dir)

    for sid in data_io.sessions:
        print(sid)
        data_io.load_session(sid, load_waveforms=False, load_pickle=True)
        data_io.dump_as_pickle()



