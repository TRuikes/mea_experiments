from pv_chip_aarhus.preprocessing.lib.filepaths import FilePaths
import pandas as pd
import numpy as np
from pv_chip_aarhus.preprocessing.params import nb_bytes_by_datapoint, data_nb_channels, data_sample_rate
from pathlib import Path
import os
import utils
import re


def extract_phy_data(filepaths: FilePaths, update=False, waveform_extraction=False, raw_data_dir=None):
    if not filepaths.proc_pp_spiketimes.exists() or update:
        _extract_spiketimes(filepaths, raw_data_dir)


def recording_onsets(filepaths: FilePaths, raw_data_dir):
    """
        Read from raw files (either links or recordings) the onsets for each rec

    Input :
        - recording_names (list) : Ordered list of raw files names to open and read length
        - path (str) : path to the directory containing  the files
        - nb_bytes_by_datapoint (int) : size in byte of each time points
        - nb_channels (int) : number of channels of the mea
    Output :
        - onsets (dict) : Dictionary of all onsets using recording_names as dict key

    Possible mistakes :
        - Wrong folders given as input
        - Mea number is wrong
    """

    with open(filepaths.proc_sc_params, 'r') as file:
        text = file.read()

    match = re.search(r'\[.*?\]', text, re.DOTALL)
    list_content = match.group(0)[1:-1]  # remove brackets
    clustered_files = [item.strip().replace('r"', '').replace('"', '').replace("'", "").replace('"', "") for item in list_content.split(',') if len(item) > 3]
    clustered_files = [Path(f).name for f in clustered_files]

    # The onset of the first recording is set to 0
    cursor = 0
    onsets = pd.DataFrame(columns=['i0', 'i1'])

    raw_data_dir = Path(raw_data_dir)
    assert raw_data_dir.exists(), f'{raw_data_dir} does not exist'

    for rec in clustered_files:
        '''
        if USE_LOCAL_DIR:
            local_name = filepaths.local_raw_dir / rec.name
        else:
            local_name = filepaths.raw_dir / rec.name
        '''

        local_name = raw_data_dir / rec

        # if this breaks, then it cant find the raw files
        assert local_name.exists(), f'{local_name}'

        # Derive name of recording
        recname = local_name.name.split('.')[0]
        onsets.at[recname, 'i0'] = np.copy(cursor)
        file_stats = os.stat(local_name)
        cursor += int(file_stats.st_size / (nb_bytes_by_datapoint * data_nb_channels))
        onsets.at[recname, 'i1'] = np.copy(cursor)
    return onsets


def _extract_spiketimes(filepaths: FilePaths, raw_data_dir):
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

    # model = load_model(filepaths.proc_sc_params)
    # model.get_cluster_channels()
    spike_clusters = np.load(filepaths.proc_phy_spike_clusters)
    spike_indices = np.load(filepaths.proc_sc_spike_times)
    cluster_overview = pd.read_csv(filepaths.proc_phy_cluster_info, sep='\t', header=0, index_col=0)
    cluster_overview = cluster_overview.query('group != "noise"')

    for cluster_i, cluster_id in enumerate(cluster_overview.index.tolist()):
        new_id = f'uid_{filepaths.sid.split("_")[0]}_{cluster_i:03d}'
        cluster_overview.at[cluster_id, 'new_id'] = new_id
        cluster_overview.at[cluster_id, 'phy_cluster_id'] = cluster_id

    spike_index_per_cluster = {}
    for cluster_id in cluster_overview.index:
        idx = np.where(spike_clusters == cluster_id)[0]
        spike_index_per_cluster[cluster_overview.loc[cluster_id, 'new_id']] = spike_indices[idx]

    onsets = recording_onsets(filepaths, raw_data_dir)

    # Group spiketimes in hierarchical dict:
    # /cluster_id /recording
    spiketimes_per_recording = {}

    for rec, rec_info in onsets.iterrows():
        spiketimes_per_recording[rec] = {}

        # Find all spike indices for this cluster
        for cluster_id, cluster_info in cluster_overview.iterrows():
            sp_idx = spike_index_per_cluster[cluster_info.new_id]

            # Find all spike indices for this cluster, in this recording
            idx = np.where((sp_idx >= rec_info.i0) & (sp_idx < rec_info.i1))[0]

            # Normalize spiketimes to onset of this recording, and convert to ms
            rec_spikes = ((sp_idx[idx] - rec_info.i0) / data_sample_rate) * 1000

            # Write the spiketimes to output dit
            spiketimes_per_recording[rec][cluster_info.new_id] = rec_spikes

    utils.store_nested_dict(filepaths.proc_pp_spiketimes, spiketimes_per_recording)

    cluster_overview.set_index(['new_id'], inplace=True)
    cluster_overview.to_csv(filepaths.proc_pp_clusterinfo)

    n_clusters = cluster_overview.shape[0]
    print(f'\textracted spike for {n_clusters} clusters!')
