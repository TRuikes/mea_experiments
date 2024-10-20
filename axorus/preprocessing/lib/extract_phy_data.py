from axorus.preprocessing.lib.filepaths import FilePaths
import pandas as pd
import numpy as np
from axorus.preprocessing.params import nb_bytes_by_datapoint, data_nb_channels, data_sample_rate
from pathlib import Path
import os
import utils



def extract_phy_data(filepaths: FilePaths, update=False):
    get_spikedata(filepaths, update=update)


def get_spikedata(filepaths: FilePaths, update):
    print(f'Extracting spiketimes')
    if filepaths.proc_pp_spiketimes.exists() and not update:
        print(f'\tspiketimes already saved!')
        return

    _extract_spiketimes(filepaths)

    return


def recording_onsets(filepaths: FilePaths):
    """
        Read from raw files (either links or recordings) the onsets for each rec

    Input :
        - recording_names (list) : Ordered list of raw files names to open and read length
        - path (str) : path to the directory containing the files
        - nb_bytes_by_datapoint (int) : size in byte of each time points
        - nb_channels (int) : number of channels of the mea
    Output :
        - onsets (dict) : Dictionary of all onsets using recording_names as dict key

    Possible mistakes :
        - Wrong folders given as input
        - Mea number is wrong
    """


    # Detect which files were clustered
    clustered_files = []

    with open(filepaths.proc_sc_params, 'r') as file:
        text = file.read()

    for line in text.split('\n'):
        if 'dat_path' in line:

            parts0 = line.split('[')
            prefix = parts0[0]
            parts1 = parts0[1].split(']')[0]
            parts2 = parts1.split(',')

            prefix += '['
            for filename in parts2:
                if len(filename) < 3:
                    continue
                f = Path(filename.split('"')[1])
                clustered_files.append(f)

    # The onset of the first recording is set to 0
    cursor = 0
    onsets = pd.DataFrame(columns=['i0', 'i1'])

    for rec in clustered_files:
        assert rec.exists()

        # Derive name of recording
        recname = rec.name.split('.')[0]

        onsets.at[recname, 'i0'] = np.copy(cursor)

        file_stats = os.stat(rec)
        cursor += int(file_stats.st_size / (nb_bytes_by_datapoint * data_nb_channels))
        onsets.at[recname, 'i1'] = np.copy(cursor)
    return onsets


def _extract_spiketimes(filepaths: FilePaths):
    """
        Read phy variables and extract the spiking times of each cluster
    Input :
        - directory (str) : phy varariables directory
    Output :
        - spike_times (dict) : Dictionnary of each cluster's spiking time, cluster_id as key and a list as value

    Possible mistakes :
        - Wrong directory
        - .npy files no longer exists

    """

    spike_clusters = np.load(filepaths.proc_phy_spike_clusters)
    spike_indices = np.load(filepaths.proc_sc_spike_times)
    cluster_overview = pd.read_csv(filepaths.proc_phy_cluster_info, sep='\t', header=0, index_col=0)
    cluster_overview = cluster_overview.query('group != "noise"')

    spike_index_per_cluster = {}
    for cluster_id in cluster_overview.index:
        idx = np.where(spike_clusters == cluster_id)[0]
        spike_index_per_cluster[cluster_id] = spike_indices[idx]

    onsets = recording_onsets(filepaths)

    # Group spiketimes in hierarchical dict:
    # /cluster_id /recording
    spiketimes_per_recording = {}

    for rec, rec_info in onsets.iterrows():
        spiketimes_per_recording[rec] = {}

        # Find all spike indices for this cluster
        for cluster_id, cluster_info in cluster_overview.iterrows():
            sp_idx = spike_index_per_cluster[cluster_id]

            # Find all spike indices for this cluster, in this recording
            idx = np.where((sp_idx >= rec_info.i0) & (sp_idx < rec_info.i1))[0]
            # Normalize spiketimes to onset of this recording, and convert to ms
            rec_spikes = ((sp_idx[idx] - rec_info.i0) / data_sample_rate) * 1000
            # Write the spiketimes to output dit
            spiketimes_per_recording[rec][f'{cluster_id}'] = rec_spikes

    utils.store_nested_dict(filepaths.proc_pp_spiketimes, spiketimes_per_recording)
    cluster_overview.to_csv(filepaths.proc_pp_clusterinfo)

    n_clusters = cluster_overview.shape[0]
    print(f'\textracted spike for {n_clusters} clusters!')


def extract_waveforms(filepaths: FilePaths, update):
    return