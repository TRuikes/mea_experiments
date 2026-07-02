from pathlib import Path
from pv_chip_aarhus.preprocessing.params import dataset_dir
import numpy as np
import pandas as pd


def get_recording_table(trigger_dir, stim_dir):
    recording_ids = [(file.name.split('_rec_')[1].split('_')[0], file.name) for file in trigger_dir.iterdir() if '_triggers.pkl' in file.name]

    stimfiles_by_prefix = {
        p.name[:2]: p
        for p in stim_dir.iterdir()
        if p.is_file()
    }

    recording_table = pd.DataFrame()
    for rec_nr, trigger_file in recording_ids:

        # If this breaks, its because the files are not exaclyt named the same
        recording_table.at[rec_nr, 'trigger_file'] = trigger_file
        date_tf, _, _, _, _, strain_tf, animal_id, ret_nr, slice_nr, _, _, recnr_tf, stimtype_tf, lasermode_tf, stimsource_tf, _  = trigger_file.split('_')
        # If this breaks, its because the files are not exaclyt named the same
        stim_file = stimfiles_by_prefix[rec_nr].name
        recnr_sf, stimtype_sf, lasermode_sf, stimsource_sf, date_sf, channel_sf, _, _, _ = stim_file.split('_')

        # Do some sanity checks
        # if this breaks, it means the filenames between trigger and recording are not 100% exactly the same
        assert recnr_tf == recnr_sf
        assert lasermode_tf == lasermode_sf
        assert stimsource_tf == stimsource_sf

        stimtype_tf = stimtype_sf.split('-')[0]
        assert stimtype_sf == stimtype_tf

        recording_table.at[rec_nr, 'stim_file'] = stim_file
        recording_table.at[rec_nr, 'lasermode'] = lasermode_sf
        recording_table.at[rec_nr, 'stimsource'] = stimsource_sf
        recording_table.at[rec_nr,  'laser_ch'] = channel_sf
        recording_table.at[rec_nr, 'varied_param'] = stimtype_sf

    return recording_table


class FilePaths:
    """
    Generate filepaths within the datasets
    Recorded and processed data is stored in a basepath named 'series 3',
    e.g. E/Photoacoustic/series 3'. Then data is stored as

    ~/SESSION_ID/
        sorting_files/
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

        stim_files/
            01_mynamedmcstimfile.dat       # 01 = RECNR
            02_mynamedmcstimfile.dat
            ...
            NN_mynamedmcstimfile.dat

        trigger_files/
            YYYYMMDD_PV_CHIP_name_rec_RECNR_morename_triggers.pkl

    """

    # Allowed names for attributes
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

            self.session_dir = Path(dataset_dir) / sid
            self.sorting_dir = self.session_dir / 'sorting_files'
            self.stim_dir = self.session_dir / 'stim_files'
            self.trigger_dir = self.session_dir / 'trigger_files'
            self.misc_dir = self.session_dir / 'misc'

            self.recording_table = get_recording_table(trigger_dir=self.trigger_dir, stim_dir=self.stim_dir)

            raw_mea_position = [f for f in self.misc_dir.iterdir() if 'MEA_position' in f.name and f.suffix == '.csv']
            assert len(raw_mea_position) == 1, f'Check MEA position files found in {self.misc_dir}'
            self.mea_position_file = raw_mea_position[0]

            if self.sorting_dir.exists():
                self.has_sorted_data = True
                self.proc_sc_amplitudes = self.sorting_dir / 'amplitudes.npy'
                self.proc_sc_channel_map = self.sorting_dir / 'channel_map.npy'
                self.proc_sc_channel_positions = self.sorting_dir / 'channel_positions.npy'
                self.proc_sc_similar_templates = self.sorting_dir / 'similar_templates.npy'
                self.proc_sc_spike_times = self.sorting_dir / 'spike_times.npy'
                self.proc_sc_template_ind = self.sorting_dir / 'template_ind.npy'
                self.proc_sc_templates = self.sorting_dir / 'templates.npy'
                self.proc_sc_whitening_mat = self.sorting_dir / 'whitening_mat.npy'
                self.proc_sc_whitening_mat_inv = self.sorting_dir / 'whitening_mat_inv.npy'
                self.proc_sc_params = self.sorting_dir / 'params.py'

                self.proc_phy_cluster_group = self.sorting_dir / 'cluster_group.tsv'

                if self.proc_phy_cluster_group.exists():
                    self.has_manual_sorted_data = True

                    self.proc_phy_cluster_info = self.sorting_dir / 'cluster_info.tsv'
                    self.proc_phy_spike_clusters = self.sorting_dir / 'spike_clusters.npy'
                    self.proc_phy_cluster_purity = self.sorting_dir / 'cluster_purity.tsv'

                else:
                    self.has_manual_sorted_data = False

            else:
                self.has_sorted_data = False

            # define trials file
            self.proc_pp_trials = self.misc_dir / 'trials.csv'

            # define spiketimes file
            self.proc_pp_spiketimes = self.misc_dir / 'spiketimes.h5'
            self.proc_pp_clusterinfo = self.misc_dir / 'cluster_info.csv'

            # Define the final dataset file
            self.dataset_file = self.dataset_out_dir / f'{self.sid}.h5'
            self.dataset_file_waveforms = self.dataset_out_dir / f'{self.sid}_waveforms.h5'


if __name__ == '__main__':
    sid = 'test_data'
    f = FilePaths(sid)
    print(f.recording_nrs)
