from axorus.preprocessing.lib.filepaths import FilePaths
import pandas as pd
import numpy as np
from axorus.preprocessing.params import nb_bytes_by_datapoint, data_nb_channels, data_sample_rate, dataset_dir, data_voltage_resolution, data_type
from pathlib import Path
import os
import utils
from tqdm import tqdm
import re


def extract_phy_data(filepaths: FilePaths, update=False):
    get_spikedata(filepaths, update=update)
    # _extract_waveforms(filepaths, update=update)


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

    if filepaths.local_raw_dir is not None:
        USE_LOCAL_DIR = True
    else:
        USE_LOCAL_DIR = False

    with open(filepaths.proc_sc_params, 'r') as file:
        text = file.read()

    match = re.search(r'\[.*?\]', text, re.DOTALL)
    list_content = match.group(0)[1:-1]  # remove brackets
    clustered_files = [item.strip().replace('r"', '').replace('"', '') for item in list_content.split(',') if len(item) > 3]
    clustered_files = [Path(f) for f in clustered_files]

    # for line in text.split('\n'):
    #     if 'dat_path' in line:
    #
    #         parts0 = line.split('[')
    #         prefix = parts0[0]
    #         parts1 = parts0[1].split(']')[0]
    #         parts2 = parts1.split(',')
    #
    #         prefix += '['
    #         for filename in parts2:
    #             if len(filename) < 3:
    #                 continue
    #             f = Path(filename.split('"')[1])
    #             clustered_files.append(f)

    # The onset of the first recording is set to 0
    cursor = 0
    onsets = pd.DataFrame(columns=['i0', 'i1'])

    for rec in clustered_files:

        if USE_LOCAL_DIR:
            local_name = filepaths.local_raw_dir / rec.name
        else:
            local_name = filepaths.raw_dir / rec.name

        assert local_name.exists(), f'{local_name}'

        # Derive name of recording
        recname = local_name.name.split('.')[0]

        onsets.at[recname, 'i0'] = np.copy(cursor)

        file_stats = os.stat(local_name)
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

    onsets = recording_onsets(filepaths)

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


def _extract_waveforms(filepaths: FilePaths, update):
    print(f'\nProcessing waveforms')
    if filepaths.proc_pp_waveforms.exists() and not update:
        print(f'\twaveforms allready extracted!')
        return

    onsets = recording_onsets(filepaths)
    spiketimes = utils.load_nested_dict(filepaths.proc_pp_spiketimes)
    cluster_overview = pd.read_csv(filepaths.proc_pp_clusterinfo,
                                   index_col=0, header=0)

    recnames = onsets.index.tolist()

    NWAVEFORMS = 1000  # nr of waveforms to extract
    NPRE = 5  # time in ms
    NPOST = 10  # time in ms
    NSAMPLES = int(((NPRE + NPOST) / 1000) * data_sample_rate)

    n_samples_pre = (NPRE / 1000) * data_sample_rate
    recording_channel_nrs = cluster_overview.ch.unique()

    cluster_waveforms = dict()

    print('\textracting waveforms...')

    for rname in recnames:
        if filepaths.local_raw_dir is not None:
            input_file = filepaths.local_raw_dir / f'{rname}.raw'
        else:
            input_file = filepaths.raw_dir / f'{rname}.raw'

        if not input_file.is_file():
            raise ValueError(f'\twaveforms: cant find file: {input_file}')

        # Open datafile
        m = np.memmap(input_file.as_posix(), dtype=data_type)

        # Add key for current recording
        cluster_waveforms[rname] = dict()

        # Load data from per channel
        for channel_nr in tqdm(recording_channel_nrs):
            channel_index = np.arange(channel_nr, m.size, data_nb_channels)

            # Find all clustres on current channel
            cluster_ids = cluster_overview.query('ch == @channel_nr').index.tolist()

            for cluster_id in cluster_ids:
                spike_times_rec = spiketimes[rname][cluster_id]

                # Draw a NWAVEFORMS random spiketimes
                idx = np.arange(0, spike_times_rec.size)
                if spike_times_rec.size > NWAVEFORMS:
                    idx = np.sort(np.random.choice(idx, NWAVEFORMS, replace=False))

                # Load data per spike time
                all_idx = []
                for si, spike_time in enumerate(spike_times_rec[idx]):
                    sref = (spike_time / 1000) * data_sample_rate
                    i0 = int(sref - n_samples_pre)
                    if i0 < 0:
                        continue
                    i1 = int(i0 + NSAMPLES)

                    all_idx.append(np.arange(i0,i1,1))
                if len(all_idx) == 0:
                    cluster_waveforms[rname][cluster_id] = np.array([])
                    continue

                try:
                    all_idx = np.hstack(all_idx)
                    wf = m[channel_index[all_idx]].copy()
                    wf = wf.reshape([idx.size, NSAMPLES])

                    wf = wf * data_voltage_resolution - 4096  # this is in uV now
                    cluster_waveforms[rname][cluster_id] = wf
                except ValueError:
                    cluster_waveforms[rname][cluster_id] = np.array([])
                    continue

    utils.store_nested_dict(filepaths.proc_pp_waveforms, cluster_waveforms)
    print(f'\tStored waveforms: {filepaths.proc_pp_waveforms}')