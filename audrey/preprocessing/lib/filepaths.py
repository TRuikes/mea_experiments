import sys
sys.path.append('.')
from pathlib import Path
from datetime import datetime
from audrey.preprocessing.params import dataset_dir
import numpy as np


def extract_date(sid: str):
    # Extract day, month, and year from the string
    sid_split = sid.split('_')[0]
    day = sid_split[4:]
    month = sid_split[2:4]
    year = sid_split[:2]

    # Convert the two-digit year to four digits
    year = '20' + year if int(year) <= 99 else '19' + year

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
            YYMMDD_SLICENR_RECNR_BLOCKER.mcd
            YYMMDD_SLICENR_RECNR_BLOCKER.raw
            YYMMDD_SLICENR_RECNR_BLOCKER_protocols.csv
            YYMMDD_SLICENR_RECNR_BLOCKER_trials.csv
    """

    # Allowed names for attributes
    blocker_names = ('noblocker', 'washout')
    slice_names = ('A', 'B', 'C', 'D')
    rec_names = ('1', '2', '3', '4', '5', '6', '7', '8', '9')

    laser_calib_path = None
    laser_calib_file = None
    laser_calib_figure_dir = None

    def __init__(self, sid=None, laser_calib_week=None, local_raw_dir=None):
        """
        session ids have shape DDMMYY_retslice
        """
        self.sid = sid
        if not Path(dataset_dir).is_dir():
            raise ValueError(f'Cannot find datasetdir: {dataset_dir}')
        self.dataset_dir = Path(dataset_dir)
        self.laser_calib_week = laser_calib_week
        self.dataset_out_dir = self.dataset_dir / 'dataset'
        self.local_raw_dir = local_raw_dir
        if local_raw_dir is not None:
            self.local_raw_dir = Path(local_raw_dir)
        else:
            self.local_raw_dir is None

        if sid is not None:
            self.date = extract_date(sid)
            if '_' in sid:
                self.slice_nr = sid.split('_')[1]
            else:
                self.slice_nr = 'A'

            self.processed_dir = self.dataset_dir / sid / 'processed'
            self.raw_dir = self.dataset_dir / sid / 'raw'
            self.csv_dir = self.dataset_dir / sid / 'csv'
            #self.raw_dir = self.dataset_dir / sid / 'sorted'

            if self.local_raw_dir is not None:
                self.blocker = detect_blocker(self.local_raw_dir)
                recording_nrs = detect_rec_nrs(self.local_raw_dir)
            else:
                self.blocker = detect_blocker(self.raw_dir)
                recording_nrs = detect_rec_nrs(self.raw_dir)
            self.recording_nrs = recording_nrs
            # self.rec_nr = detect_rec_nr(self.raw_dir)

            # define raw files
            rec_code = f'{str(self.date.year)[-2:]}{self.date.month}{self.date.day:02d}_{self.slice_nr}'
            self.raw_mcds = [f for f in self.raw_dir.iterdir() if rec_code in f.name and f.suffix == '.mcd']
            self.raw_raws = [f for f in self.raw_dir.iterdir() if rec_code in f.name and f.suffix == '.raw']

            raw_trials_tmp = [f for f in self.csv_dir.iterdir() if '_trials.csv' in f.name]
            raw_trials = [f for f in raw_trials_tmp if '~lock' not in f.name]
            raw_t_nrs = [int(f.name.split('_')[2]) for f in raw_trials]
            sort_idx = np.argsort(raw_t_nrs)
            self.raw_trials = [raw_trials[s] for s in sort_idx]

            self.raw_protocols = self.raw_dir / (rec_code + '_protocols.csv')

            # raw_mea_position = self.csv_dir / f'{self.date.year}-{self.date.month}-{self.date.day}_MEA_position.csv'
            # if not raw_mea_position.exists():
            self.raw_mea_position = self.csv_dir / f'{self.date.year}-{self.date.month:02.0f}-{self.date.day}_MEA_position.csv'
            #  = raw_mea_position

            # define processed files
            self.sorted_dir = self.dataset_dir / sid / 'sorted'

            self.gui_dir = None
            # Detect path with manual sorting output
            for path in self.sorted_dir.iterdir():
                if path.is_dir():
                    for path_2 in path.iterdir():
                        if '.GUI' in path_2.name:
                            self.gui_dir = path_2

            assert self.gui_dir is not None


            if self.sorted_dir.exists():
                self.has_sorted_data = True
                self.proc_sc_amplitudes = self.gui_dir / 'amplitudes.npy'
                self.proc_sc_channel_map = self.gui_dir / 'channel_map.npy'
                self.proc_sc_channel_positions = self.gui_dir / 'channel_positions.npy'
                self.proc_sc_similar_templates = self.gui_dir / 'similar_templates.npy'
                self.proc_sc_spike_times = self.gui_dir / 'spike_times.npy'
                self.proc_sc_template_ind = self.gui_dir / 'template_ind.npy'
                self.proc_sc_templates = self.gui_dir / 'templates.npy'
                self.proc_sc_whitening_mat = self.gui_dir / 'whitening_mat.npy'
                self.proc_sc_whitening_mat_inv = self.gui_dir / 'whitening_mat_inv.npy'
                self.proc_sc_params = self.gui_dir / 'params.py'

                self.proc_phy_cluster_group = self.gui_dir / 'cluster_group.tsv'

                if self.proc_phy_cluster_group.exists():
                    self.has_manual_sorted_data = True

                    self.proc_phy_cluster_info = self.gui_dir / 'cluster_info.tsv'
                    self.proc_phy_spike_clusters = self.gui_dir / 'spike_clusters.npy'
                    self.proc_phy_cluster_purity = self.gui_dir / 'cluster_purity.tsv'

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
            self.proc_pp_artefact_positions = self.processed_dir / 'artefact_positions.csv'

            # define trials file
            self.proc_pp_trials = self.processed_dir / 'trials.csv'

            # Define the final dataset file
            self.dataset_file = self.dataset_out_dir / f'{self.sid}.h5'
            self.dataset_file_waveforms = self.dataset_out_dir / f'{self.sid}_waveforms.h5'

            self.get_recording_names_from_rawfiles()

        self.set_laser_calib_files()

    def set_laser_calib_files(self):
        if self.laser_calib_week is None:
            return
        else:
            self.laser_calib_path = self.dataset_dir / 'laser_calibration' / self.laser_calib_week
            self.laser_calib_file = self.laser_calib_path / f'laser_calibration_{self.laser_calib_week}.csv'
            self.laser_calib_power_file = self.laser_calib_path / f'laser_calibration_{self.laser_calib_week}_power.csv'
            self.laser_calib_figure_dir = self.laser_calib_path / 'figures'

    def check_data(self):
        print(f'Checking if all data is available for {self.sid}:')
        if not self.processed_dir.exists():
            self.processed_dir.mkdir(parents=True)

        assert self.raw_dir.exists(), f'{self.raw_dir} does not exist'
        assert self.blocker in self.blocker_names, f'{self.blocker}'
        assert self.slice_nr in self.slice_names

        assert self.raw_mea_position.exists(), f'did not find: {self.raw_mea_position}'

        assert len(self.raw_trials) > 0, f'no trial files found'
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
        readdir = self.local_raw_dir if self.local_raw_dir is not None else self.raw_dir

        recording_names = [f.name.split('.')[0] for f in readdir.iterdir() if 'raw' in f.name]

        if len(recording_names) == 0:
            # check if there are rawfiles
            recording_names = [f.name.split('.')[0] for f in readdir.iterdir() if
                                    'raw' in f.name]

        if len(recording_names) == 0:
            raise ValueError('no recordings found')

        recnames_int = [int(r.split('_')[2]) for r in recording_names]
        self.recording_names = [recording_names[i] for i in np.argsort(recnames_int)]


if __name__ == '__main__':
    sid = '250904_A'
    f = FilePaths(sid, local_raw_dir=None)
    f.check_data()
