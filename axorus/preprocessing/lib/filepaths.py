from pathlib import Path
from datetime import datetime
from axorus.preprocessing.params import dataset_dir
import numpy as np


def extract_date(sid: str):
    # Extract day, month, and year from the string
    sid_split = sid.split('_')[0]
    day = sid_split[:2]
    month = sid_split[2:4]
    year = sid_split[4:]

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


def detect_rec_nr(raw_dir: Path):
    for f in raw_dir.iterdir():
        if f.suffix == '.raw':
            rec_nr = f.name.split('_')[2].split('.')[0]
            return rec_nr

    raise FileNotFoundError(f'Cant find rawfile in {raw_dir}')


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
    blocker_names = ('noblocker')
    slice_names = ('A', 'B', 'C', 'D')
    rec_names = ('1', '2', '3', '4', '5', '6', '7', '8', '9')

    def __init__(self, sid: str):
        """
        session ids have shape DDMMYY_retslice
        """
        self.sid = sid
        self.dataset_dir = Path(dataset_dir)
        self.date = extract_date(sid)
        self.slice_nr = sid.split('_')[1]

        self.processed_dir = self.dataset_dir / sid / 'processed'
        self.raw_dir = self.dataset_dir / sid / 'raw'
        self.blocker = detect_blocker(self.raw_dir)
        self.rec_nr = detect_rec_nr(self.raw_dir)

        # define raw files
        rec_code = f'{str(self.date.year)[-2:]}{self.date.month}{self.date.day}_{self.slice_nr}_{self.rec_nr}_{self.blocker}'
        self.raw_mcd = self.raw_dir / (rec_code + '.mcd')
        self.raw_raw = self.raw_dir / (rec_code + '.raw')
        self.raw_trials = self.raw_dir / (rec_code + '_trials.csv')
        self.raw_protocols = self.raw_dir / (rec_code + '_protocols.csv')

        # define processed files
        self.sorted_dir = self.processed_dir / 'sorted'
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

        self.get_recording_names_from_rawfiles()

    def check_data(self):
        print(f'Checking if all data is available for {self.sid}:')
        assert self.processed_dir.exists()
        assert self.raw_dir.exists()
        assert self.blocker in self.blocker_names
        assert self.slice_nr in self.slice_names
        assert self.rec_nr in self.rec_names

        # assert self.raw_mcd.exists(), f'{self.raw_mcd} does not exist'
        assert self.raw_raw.exists(), f'{self.raw_raw} does not exist'
        assert self.raw_trials.exists()
        assert self.raw_protocols.exists()

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
        recording_names = [f.name.split('.')[0] for f in self.raw_dir.iterdir() if 'mcd' in f.name]

        if len(recording_names) == 0:
            # check if there are rawfiles
            if (self.recording_dir / 'raw').is_dir():
                recording_names = [f.name.split('.')[0] for f in self.raw_dir.iterdir() if
                                        'raw' in f.name]
            else:
                raise FileNotFoundError

        recnames_int = [int(r.split('_')[2]) for r in recording_names]
        self.recording_names = [recording_names[i] for i in np.argsort(recnames_int)]


if __name__ == '__main__':
    sid = '161024_A'
    f = FilePaths(sid)
    f.check_data()