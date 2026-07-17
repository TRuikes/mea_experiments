from pathlib import Path
import h5py
import pandas as pd
import threading
import pickle
import numpy as np
from typing import List, no_type_check


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

        date = session_id.split()[0].split('-')
        date_short = f'{date[0][2:]}{date[1]}{date[2]}_{session_id.split()[-1]}'
        self.sid_short = date_short
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
            burst_rows = []


            # --- Load top-level cluster metadata ---
            if "clusters/metadata" in f:
                cluster_df = hdf5_structured_array_to_df(f["clusters/metadata"])

                # Tiny pathc to reach Audrey's data
                if 'Audrey' in session_id:
                    r0 = list(f['recordings'].keys())[0]
                    cluster_ids = list(f['recordings'][r0]['clusters'].keys())
                    cluster_df['new_id'] = cluster_ids

                cluster_df.set_index('new_id', inplace=True)

            # --- Load recordings ---
            for rec_id in f["recordings"].keys():
                rec_ids.append(rec_id)
                rec_grp = f[f"recordings/{rec_id}"]

                triggers_df = hdf5_structured_array_to_df(rec_grp["triggers"])

                trial_info_df = hdf5_structured_array_to_df(rec_grp["trial_info"])
                trial_info_df.set_index('train_id', inplace=True)

                # Convert object columns to strings
                for col in trial_info_df.select_dtypes([np.object_]):
                    trial_info_df[col] = trial_info_df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else str(x))


                # Build burst data using a list of dictionaries
                for i, row in triggers_df.iterrows():
                    new_id = f"{rec_id}-{i}"

                    # Start a dictionary for this row
                    row_dict = {
                        'row_id': new_id,  # Keep track of your index
                        'rec_id': rec_id,
                        'burst_id': i
                    }

                    # Add trigger_df keys
                    for k in row.index:
                        row_dict[k] = row[k]

                    # Add trial info for matching train_id
                    train_id = row["train_id"]
                    trial_row = trial_info_df.loc[train_id]
                    for k, v in trial_row.items():
                        if isinstance(v, str) and not isinstance(v, bool):
                            if v.lower() == 'false':
                                v = False
                            elif v.lower() == 'true':
                                v = True
                            elif v.lower() in ['nan', 'none', '']:
                                v = np.nan
                        row_dict[k] = v

                    burst_rows.append(row_dict)

                # Load per-recording clusters
                spiketimes[rec_id] = {}
                waveforms[rec_id] = {}
                if "clusters" in rec_grp:
                    for cluster_id in rec_grp["clusters"].keys():
                        cluster_rec_grp = rec_grp[f"clusters/{cluster_id}"]
                        spiketimes[rec_id][cluster_id] = cluster_rec_grp["spiketimes"][()]
                        if load_waveforms and "waveforms" in cluster_rec_grp:
                            waveforms[rec_id][cluster_id] = cluster_rec_grp["waveforms"][()]


        # Create the final DataFrame all at once
        burst_df = pd.DataFrame(burst_rows)
        burst_df.set_index('row_id', inplace=True)


        # Store to instance
        self.burst_df = burst_df
        self.cluster_df = cluster_df
        self.spiketimes = spiketimes
        self.waveforms = waveforms if load_waveforms else None
        self.recording_ids = rec_ids

        train_rows = []

        for tid, tdf in burst_df.groupby("train_id"):
            # Start a dictionary for this specific train_id row
            row_dict = {"train_id": tid}

            for c in tdf.columns:
                if c in ['burst_id'] or 'burst_onset' in c or 'burst_offset' in c:
                    continue
                assert len(tdf[c].unique()) == 1, f"Column '{c}' has non-unique values for train_id {tid}"

                val = tdf.iloc[0][c]
                if isinstance(val, bool):
                    val = float(val)  # True → 1.0, False → 0.0

                # Save to the row dictionary instead of the DataFrame
                row_dict[c] = val

            train_rows.append(row_dict)

        # Build the DataFrame all at once at the very end
        train_df = pd.DataFrame(train_rows)
        if not train_df.empty:
            train_df.set_index("train_id", inplace=True)



        for i, r in train_df.iterrows():
            if 'sequence_name' in r.keys():
                train_df.at[i, 'protocol_name'] = r['sequence_name']
            elif 'protocol_name' in r.keys():
                continue
            else:
                train_df.at[i, 'protocol_name'] = r['recording_name']

            if 'dac_voltage' in r.keys():
                train_df.at[i, 'laser_power'] = r['dac_voltage']

        self.train_df = train_df

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
    from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis

    data_io = DataIO(dataset_dir)
    session_id = '2026-07-08 rat LE 3322 Mekano6 A'

    figure_dir_analysis = figure_dir_analysis / session_id
    print(session_id)
    data_io.load_session(session_id, load_pickle=False, load_waveforms=False)
    for r in data_io.recording_ids:
        print(r)