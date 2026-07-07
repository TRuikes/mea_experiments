from pathlib import Path
import h5py
import pandas as pd
import threading
import pickle
import numpy as np
from typing import List, no_type_check
import numpy as np
import pandas as pd


def hdf5_structured_array_to_df(arr: np.ndarray) -> pd.DataFrame:
    """Reverse engineers a NumPy structured array back into a pandas DataFrame.

    Handles DTypePromotionError by safely converting string columns to object arrays
    before inserting np.nan values.
    """
    data = {}

    for name in arr.dtype.names:
        col_data = arr[name]
        kind = col_data.dtype.kind

        if kind in ("S", "V"):  # Byte strings / structured void
            # 1. Decode bytes to utf-8 strings
            decoded_strings = np.char.decode(col_data, "utf-8")

            # 2. Convert to an 'object' array so it can cleanly hold both strings and np.nan
            obj_strings = decoded_strings.astype(object)

            # 3. Replace empty strings with np.nan safely
            data[name] = np.where(obj_strings == "", np.nan, obj_strings)

        elif kind == "f":  # Floats
            float_data = col_data.astype(np.float64)
            data[name] = np.where(float_data == -99.0, np.nan, float_data)

        else:  # Catch-all for integers, booleans, etc.
            data[name] = col_data

    return pd.DataFrame(data)
class DataIO:
    sessions = []
    recording_ids: List[str] = []
    burst_df = pd.DataFrame()
    cluster_df = pd.DataFrame()
    spiketimes: dict[str, dict[str, np.ndarray]] = {} # type: ignore
    waveforms = {}
    cluster_ids = []
    sid_short = None

    def __init__(self, datadir: Path):
        self.datadir = Path(datadir)
        self.detect_sessions()
        self.lock = threading.Lock()
        self.is_locked = False

    def detect_sessions(self):
        self.sessions = [f.name.split('.')[0] for f in self.datadir.iterdir() if f.suffix == ".h5"]

    @no_type_check
    def load_session(self, session_id: str, load_pickle: bool=True):
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
        spiketimes = {}
        rec_ids = []

        # Determine which HDF5 file to open
        readfile = self.datadir / f'{session_id}.h5'

        burst_dataframes = []

        with h5py.File(readfile.as_posix(), 'r') as f:
            # --- Load top-level cluster metadata ---
            cluster_df = hdf5_structured_array_to_df(f['cluster_df'])
            recording_df = hdf5_structured_array_to_df(f['recording_df'])
            trial_df = hdf5_structured_array_to_df(f['trial_df'])


            # --- Load recordings ---
            for rec_id in f["recordings"].keys():
                rec_ids.append(rec_id)
                rec_grp = f[f"recordings/{rec_id}"]

                # Load triggers → burst_df
                triggers_df = hdf5_structured_array_to_df(rec_grp["triggers_df"])
                triggers_df['rec_id'] = rec_id
                burst_dataframes.append(triggers_df)

                # Load per-recording clusters
                spiketimes[rec_id] = {}
                if "clusters" in rec_grp:
                    for cluster_id in rec_grp["clusters"].keys():
                        cluster_rec_grp = rec_grp[f"clusters/{cluster_id}"]
                        spiketimes[rec_id][cluster_id] = cluster_rec_grp["spiketimes"][()]

        # Store to instance
        self.recording_df = recording_df
        self.cluster_df = cluster_df
        self.trial_df = trial_df
        self.burst_df = pd.concat(burst_dataframes, ignore_index=True)
        self.spiketimes = spiketimes
        self.cluster_ids = self.cluster_df.index.values

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
            cluster_ids=self.cluster_ids,
            train_df=self.train_df,
        )
        with open(self.pickle_file.as_posix(), 'wb') as f:
            pickle.dump(data_to_pickle, f)

    def load_pickle(self):
        with open(self.pickle_file, 'rb') as f:
            loaded_instance = pickle.load(f)
            for k, v in loaded_instance.items():
                self.__setattr__(k, v)

if __name__ == "__main__":
    from pv_chip_aarhus.analysis.analysis_params import dataset_dir, figure_dir_analysis

    data_io = DataIO(dataset_dir)
    session_id = 'test_data'

    figure_dir_analysis = figure_dir_analysis / session_id
    print(session_id)
    data_io.load_session(session_id, load_pickle=False)