from pathlib import Path
import h5py
import pandas as pd
import threading
import pickle
import numpy as np
from typing import List, no_type_check, Any

class DataIO:
    sessions = []
    recording_ids: List[str] = []
    burst_df = pd.DataFrame()
    cluster_df = pd.DataFrame()
    spiketimes: dict[str, dict[str, np.ndarray]] = {} # type: ignore
    waveforms = {}


    def __init__(self, datadir: Path):
        self.datadir = datadir
        self.detect_sessions()
        self.lock = threading.Lock()
        self.is_locked = False

    def detect_sessions(self):
        self.sessions = [f.name.split('.')[0] for f in self.datadir.iterdir() if f.suffix == ".h5"]

    @no_type_check
    def load_session(self, session_id: str, load_waveforms: bool=False, load_pickle: bool=True): 
        """_summary_
        Load a session from the dataset into the class
        Args:
            session_id (str): unique session id to load
            load_waveforms (bool, optional): set to true to also load waveforms. Defaults to False.
            load_pickle (bool, optional): use a pickle version of the dataset file for faster loading. Defaults to True.
        """        
        
        assert session_id in self.sessions, f'{session_id} not in {self.sessions}'
        self.session_id: str = session_id
        self.pickle_file = self.datadir / f'{session_id}.pickle'

        if load_pickle and self.pickle_file.is_file():
            print(f'Loading pickled data (not from h5 file)')
            self.load_pickle()
            return

        # Reset data
        burst_df = pd.DataFrame()
        cluster_df = pd.DataFrame()
        spiketimes = {}
        waveforms = {}
        rec_ids = []

        # Determine which HDF5 file to open
        readfile = self.datadir / (f'{session_id}_waveforms.h5' if load_waveforms else f'{session_id}.h5')

        with h5py.File(readfile.as_posix(), 'r') as f:  
            # --- Load top-level cluster metadata ---
            if "clusters/metadata" in f:
                cluster_meta = f["clusters/metadata"]
                cluster_dtype = cluster_meta.dtype  

                # Extract index field first
                index_field = cluster_dtype.names[0]  
                idx_array = cluster_meta[index_field][()] 
                # Decode bytes if necessary
                idx_array = [x.decode() if isinstance(x, bytes) else x for x in idx_array]

                # Extract remaining columns
                data_dict = {}
                for name in cluster_dtype.names[1:]:
                    col_data = cluster_meta[name][()]
                    if cluster_dtype[name].kind == 'S':  # bytes -> str
                        col_data = [x.decode() if isinstance(x, bytes) else x for x in col_data]
                    data_dict[name] = col_data

                # Reconstruct DataFrame with original index
                cluster_df = pd.DataFrame(data_dict, index=idx_array)

            # --- Load recordings ---
            for rec_id in f["recordings"].keys():
                rec_ids.append(rec_id)
                rec_grp = f[f"recordings/{rec_id}"]

                # Load triggers → burst_df
                triggers_ds = rec_grp["triggers"]
                triggers_df = pd.DataFrame({name: triggers_ds[name][()] for name in triggers_ds.dtype.names})

                # Convert train_id from bytes to string
                if "train_id" in triggers_df.columns:
                    triggers_df["train_id"] = triggers_df["train_id"].apply(lambda x: x.decode() if isinstance(x, bytes) else str(x))

                # Merge with trial info for burst_df
                trial_info_ds = rec_grp["trial_info"]
                trial_info_df = pd.DataFrame({name: trial_info_ds[name][()] for name in trial_info_ds.dtype.names})
                trial_info_df["train_id"] = trial_info_df["train_id"].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
                trial_info_df.set_index('train_id', inplace=True)

                # Convert object columns to strings
                for col in trial_info_df.select_dtypes([np.object_]):
                    trial_info_df[col] = trial_info_df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else str(x))

                # Build burst_df: one row per train_id / burst
                for i, row in triggers_df.iterrows():
                    new_id = f"{rec_id}-{i}"
                    burst_df.at[new_id, 'rec_id'] = rec_id
                    burst_df.at[new_id, 'burst_id'] = i
                    for k in row.index:
                        burst_df.at[new_id, k] = row[k]

                    # Add trial info for matching train_id
                    train_id = row["train_id"]
                    trial_row = trial_info_df.loc[train_id] 
                    for k, v in trial_row.items():
                        burst_df.at[new_id, k] = v

                # Load per-recording clusters
                spiketimes[rec_id] = {}
                waveforms[rec_id] = {}
                if "clusters" in rec_grp:
                    for cluster_id in rec_grp["clusters"].keys():
                        cluster_rec_grp = rec_grp[f"clusters/{cluster_id}"]
                        spiketimes[rec_id][cluster_id] = cluster_rec_grp["spiketimes"][()]
                        if load_waveforms and "waveforms" in cluster_rec_grp:
                            waveforms[rec_id][cluster_id] = cluster_rec_grp["waveforms"][()]

        # Store to instance
        self.burst_df = burst_df
        self.cluster_df = cluster_df
        self.spiketimes = spiketimes
        self.waveforms = waveforms if load_waveforms else None
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
        with open(self.pickle_file.as_posix(), 'wb') as f:
            pickle.dump(data_to_pickle, f)

    def load_pickle(self):
        with open(self.pickle_file, 'rb') as f:
            loaded_instance = pickle.load(f)
            for k, v in loaded_instance.items():
                self.__setattr__(k, v)