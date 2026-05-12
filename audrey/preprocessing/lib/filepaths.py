from pathlib import Path
from datetime import datetime
from sonogenetics.preprocessing.params import dataset_dir
import pandas as pd
import numpy as np


def extract_date(sid: str):
    # Extract day, month, and year from the string
    # sid_split = sid.split()[0].split('-')
    # day = sid_split[2]
    # month = sid_split[1]
    # year = sid_split[0]
    day = sid[4:6]
    month = sid[2:4]
    year = sid[:2]
    if len(year) == 2:
        year = '20' + year

    # Format the date as a datetime object
    date_str = f"{day}-{month}-{year}"
    date_obj = datetime.strptime(date_str, "%d-%m-%Y")

    return date_obj


def detect_blocker(raw_dir: Path):
    for f in raw_dir.iterdir():
        if f.suffix == '.raw':
            blocker = f.name.split('_')[3].split('.')[0]
            return blocker

    raise FileNotFoundError(f'Cant find rawfile in {raw_dir}')


def detect_rec_nrs(raw_dir: Path):
    rec_nrs = []
    for f in raw_dir.iterdir():
        if f.suffix == '.raw':
            rec_nrs.append(f.name.split('_')[2].split('.')[0])
    return rec_nrs


class FilePaths:
    """
    Generate filepaths within the datasets
    Recorded and processed data is stored in a basepath named 'series 3',
    e.g. E/Photoacoustic/series 3'. Then data is stored as

    ~/SESSION_ID/
        processed/
            sorted/
                amplitudes.npy
                channel_map.npy
                channel_positions.npy
                channel_shanks.npy
                cluster_group.tsv
                cluster_purity.tsv
                params.py
                phy.txt
                similar_templates.npy
                spike_times.mpy
                template_ind.npy
                whitening_mat.npy
                whitening_mat_inv.npy
        
        raw/
            YYYY-MM-DD_MEA_position.csv
            YYYY-MM-DD_onda_laser_calibration.json
            probefile.prb

            rec_1_STIMDETAILS.raw
            rec_2_STIMDETAILS.raw
            ...
            rec_N_STIMDETAILS.raw

            rec_1_STIMDETAILS_trials.csv
            rec_2_STIMDETAILS_trials.csv
            ...
            rec_N_STIMDETAILS_trials.csv

            PROTOCOL_1.py
            PROTOCOL_2.py
        
    """

    # Allowed names for attributes
    blocker_names = ('noblocker', 'washout', 'cppcnqx')
    slice_names = ('A', 'B', 'C', 'D')
    rec_nrs = ('1', '2', '3', '4', '5', '6', '7', '8', '9')

    def __init__(self, sid=None):
        """
        session ids have shape DDMMYY_retslice
        """
        self.sid = sid
        if not Path(dataset_dir).is_dir():
            raise ValueError(f'Cannot find datasetdir: {dataset_dir}')
        self.dataset_dir = Path(dataset_dir)
        self.dataset_out_dir = self.dataset_dir / 'dataset'

        if sid is not None:
            self.date = extract_date(sid)

            self.processed_dir = self.dataset_dir / sid / 'processed'
            self.raw_dir = self.dataset_dir / sid / 'raw'
            self.csv_dir = self.dataset_dir / sid / 'csv'

            self.recording_nrs = detect_rec_nrs(self.raw_dir)

            # detect raw files
            self.raw_mcds = [f for f in self.raw_dir.iterdir() if f.suffix == '.mcd']
            self.raw_raws = [f for f in self.raw_dir.iterdir() if f.suffix == '.raw']
            #self.raw_csvs = [f for f in self.csv_dir.iterdir() if f.suffix == '.csv']

            # raw_trials = [f for f in self.csv_dir.iterdir() if '_trials.csv' in f.name and '~lock' not in f.name]

            csv_trials_file = next(f for f in self.csv_dir.iterdir() if '_trials.csv' in f.name and '~lock' not in f.name)
            csv_trials = pd.read_csv(csv_trials_file)            

            csv_t_nrs = csv_trials['Recording Number'].unique().astype(int).tolist()
            # raw_t_nrs = [int(f.name.split('_')[2]) for f in raw_trials]
            sort_idx = np.argsort(csv_t_nrs)
            # self.raw_trials = [raw_trials[s] for s in sort_idx]
            self.csv_trials = csv_trials_file

            # raw_mea_position = [f for f in self.raw_dir.iterdir() if 'MEA_position' in f.name and f.suffix == '.csv']
            # json_mea_position = [f for f in self.csv_dir.iterdir() if 'dmd_position' in f.name and f.suffix == '.json']
            # assert len(json_mea_position) == 1, f'Check MEA position files found in {self.raw_dir}'
            # self.mea_position_file = json_mea_position[0]
            csv_mea_position = next(f for f in self.csv_dir.iterdir() if 'dmd_position' in f.name and f.suffix == '.csv')
            assert csv_mea_position.exists(), 'No dmd_position_calibration file'
            self.mea_position_file = csv_mea_position
            

            # files = [f for f in self.raw_dir.iterdir() if f.suffix == '.json' and 'laser_calibration' in f.name]
            # assert len(files) == 1, f'Check laser calibration files found in {self.raw_dir}'
            # self.laser_calibration_file = files[0]

            # define processed files
            # self.sorted_dir = self.dataset_dir / sid / 'processed' / 'sorted'
            # self.gui_dir = self.sorted_dir / f'{sid}_001_noblocker_checkerboard_30sq20px' / f'{sid}_001_noblocker_checkerboard_30sq20px.GUI'
            self.sorted_dir = self.dataset_dir / sid / 'sorted'/ f'{sid}_001_noblocker_light_SWN_acclim' / f'{sid}_001_noblocker_light_SWN_acclim.GUI'
            test = self.sorted_dir.exists()

            if self.sorted_dir.exists():
                self.has_sorted_data = True
                self.proc_sc_amplitudes = self.sorted_dir / 'amplitudes.npy'
                self.proc_sc_channel_map = self.sorted_dir / 'channel_map.npy'
                self.proc_sc_channel_positions = self.sorted_dir / 'channel_positions.npy'
                self.proc_sc_similar_templates = self.sorted_dir / 'similar_templates.npy'
                self.proc_sc_spike_times = self.sorted_dir / 'spike_times.npy'
                self.proc_sc_template_ind = self.sorted_dir / 'template_ind.npy'
                self.proc_sc_templates = self.sorted_dir / 'templates.npy'
                self.proc_sc_whitening_mat = self.sorted_dir / 'whitening_mat.npy'
                self.proc_sc_whitening_mat_inv = self.sorted_dir / 'whitening_mat_inv.npy'
                self.proc_sc_params = self.sorted_dir / 'params.py'

                self.proc_phy_cluster_group = self.sorted_dir / 'cluster_group.tsv'

                if self.proc_phy_cluster_group.exists():
                    self.has_manual_sorted_data = True

                    self.proc_phy_cluster_info = self.sorted_dir / 'cluster_info.tsv'
                    self.proc_phy_spike_clusters = self.sorted_dir / 'spike_clusters.npy'
                    self.proc_phy_cluster_purity = self.sorted_dir / 'cluster_purity.tsv'

                else:
                    self.has_manual_sorted_data = False

            else:
                self.has_sorted_data = False

            # define pipeline extracted files
            self.proc_pp_triggers = self.processed_dir / 'triggers.h5'
            self.proc_pp_spiketimes = self.processed_dir / 'spiketimes.h5'
            self.proc_pp_clusterinfo = self.processed_dir / 'cluster_info.csv'
            self.proc_pp_waveforms = self.processed_dir / 'waveforms.h5'
            self.proc_pp_figure_output = self.processed_dir / 'figures'

            # define trials file
            self.proc_pp_trials = self.processed_dir / 'trials.csv'

            # Define the final dataset file
            self.dataset_file = self.dataset_out_dir / f'{self.sid}.h5'
            self.dataset_file_waveforms = self.dataset_out_dir / f'{self.sid}_waveforms.h5'

            self.get_recording_names_from_rawfiles()


    def check_data(self):
        print(f'Checking if all data is available for {self.sid}:')
        if not self.processed_dir.exists():
            self.processed_dir.mkdir(parents=True)
        assert self.processed_dir.exists()
        assert self.raw_dir.exists()
        # assert self.slice_nr in self.slice_names

        assert self.csv_trials.exists(), f'no trial files found'
        # assert self.raw_mcd.exists(), f'{self.raw_mcd} does not exist'
        for f in self.raw_raws:
            assert f.exists()

        print(f'\tall raw data is there!')

        if self.has_sorted_data:
            assert self.proc_sc_amplitudes.exists()
            assert self.proc_sc_channel_map.exists()
            assert self.proc_sc_channel_positions.exists()
            assert self.proc_sc_similar_templates.exists()
            assert self.proc_sc_spike_times.exists()
            assert self.proc_sc_template_ind.exists()
            assert self.proc_sc_templates.exists()
            assert self.proc_sc_whitening_mat.exists()
            assert self.proc_sc_whitening_mat_inv.exists()

            print(f'\thas sorted data!')

            if self.has_manual_sorted_data:
                assert self.proc_phy_cluster_info.exists()
                assert self.proc_phy_spike_clusters.exists()
                assert self.proc_phy_cluster_purity.exists()
                assert self.proc_phy_cluster_group.exists()

                print(f'\thas manual clustered data!')

            else:
                print(f'\tdoes not have manual clustered data')

        else:
            print('\tdoes not have sorted data...')

    def get_recording_names_from_rawfiles(self):
        recording_names = [f.name.split('.')[0] for f in self.raw_dir.iterdir() if 'raw' in f.name]

        if len(recording_names) == 0:
            # check if there are rawfiles
            recording_names = [f.name.split('.')[0] for f in self.raw_dir.iterdir() if
                                    'raw' in f.name]

        if len(recording_names) == 0:
            raise ValueError('no recordings found')

        recnames_int = [int(r.split('_')[2]) for r in recording_names]
        self.recording_names = [recording_names[i] for i in np.argsort(recnames_int)]


if __name__ == '__main__':
    sid = '260424_A'
    f = FilePaths(sid)
    f.check_data()
