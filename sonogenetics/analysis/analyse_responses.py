import sys
from pathlib import Path
sys.path.append('.')
current_dir = Path().resolve()
sys.path.append(current_dir.as_posix().split('mea_experiments')[0] + 'mea_experiments')

import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import bootstrap
from scipy.signal import medfilt
from typing import List, Tuple, cast, Any, Dict, Union
from pathlib import Path

import utils
from sonogenetics.analysis.lib.analysis_params import dataset_dir
from sonogenetics.analysis.data_list import data_list
from sonogenetics.analysis.lib.data_io import DataIO
from sonogenetics.analysis.lib.poisson_rate_estimation import detect_significant_modulation_poisson, PoissonOutput

from sonogenetics.analysis.lib.bootstrap import BootstrapOutput
import scipy.stats as stats
import matplotlib.pyplot as plt



DEBUG = False


def main():
    """
    Main handles
    """
    for session_id in data_list:
        data_io = DataIO(dataset_dir)

        # session_id = data_io.sessions[0]
        print(f'Loading data: {session_id}')
        data_io.load_session(session_id, load_waveforms=False, load_pickle=False)  # type: ignore
        data_io.dump_as_pickle()

        data_io.lock_modification()

        # Analyse the cell responses following the triggers
        analyse_responses(data_io, dataset_dir / 'bootstrapped', overwrite=True)

        data_io.unlock_modification()

        # Gather all the response statistics into a single table
        gather_cluster_responses(data_io, dataset_dir / 'bootstrapped', dataset_dir / f'{data_io.session_id}_cells.csv')

        print('Done')


def gather_cluster_responses(data_io: DataIO, bootstrap_dir: Path, savename: Path):
    """
    gathers output from detect_significant_responses into a single dataframe`
    """

    # Names to store into cells dataframe

    single_value_stats = [
        'baseline_firing_rate_mean',
        # 'baseline_firing_rate_max',
        'is_excited',
        'excitation_max_fr',
        'excitation_start',
        'excitation_duration',
        'excitation_end',
        'is_inhibited',
        'inhibition_min_fr',
        'inhibition_start',
        'inhibition_duration',
        'inhibition_end',
        # 'mean_response_if_not_sig',
        # 'max_response_if_not_sig',
    ]
    # # Setup pandas dataframe
    # columns: List[Tuple[str, str]] = []
    # for train_id in data_io.train_df.index.values:
    #     for n in single_value_stats:
    #         columns.append((train_id,n))
    # multi_index = pd.MultiIndex.from_tuples(columns)
    #
    # cell_responses = pd.DataFrame(index=data_io.cluster_df.index.values, columns=multi_index)

    # for cluster_id in tqdm(cell_responses.index.values):
    #     loadname = bootstrap_dir / f'bootstrap_{cluster_id}.pkl'
    #
    #     if not loadname.exists():
    #         raise ValueError(f'loadname does not exist: {loadname}')
    #
    #     data: dict[str, BootstrapOutput] = utils.load_obj(loadname)  # type: ignore
    #
    #     cluster_x = cast(float, data_io.cluster_df.loc[cluster_id, 'cluster_x'])
    #     cluster_y = cast(float, data_io.cluster_df.loc[cluster_id, 'cluster_y'])
    #
    #
    #     for tid, tdata in data.items():
    #         laser_x: float = cast(float, data_io.train_df.loc[tid, 'laser_x'])
    #         laser_y: float = cast(float, data_io.train_df.loc[tid, 'laser_y'])
    #
    #         d = np.sqrt((cluster_x - laser_x)**2 + (cluster_y - laser_y)**2)
    #         cell_responses.at[cluster_id, (tid, 'laser_distance')] = d
    #
    #         for n in single_value_stats:
    #             cell_responses.at[cluster_id, (tid, n)] = data[tid].get(n)

    # ==========================================
    # 1. SETUP MULTIINDEX & INITIALIZE DATAFRAME
    # ==========================================
    # Get the list of cluster IDs for rows
    cluster_ids = data_io.cluster_df.index.values

    # Initialize your list of column tuples
    columns: List[Tuple[str, str]] = []
    for train_id in data_io.train_df.index.values:
        # Include your laser distance metric first
        columns.append((train_id, 'laser_distance'))
        # Include all other single-value statistics
        for n in single_value_stats:
            columns.append((train_id, n))

    # Create the MultiIndex from the list of tuples
    multi_index = pd.MultiIndex.from_tuples(columns, names=['tid', 'metric'])

    # Initialize the empty DataFrame with NaNs
    cell_responses = pd.DataFrame(
        index=cluster_ids,
        columns=multi_index,
        dtype=float
    )

    # ==========================================
    # 2. CREATE FAST INDEX MAPS & PRE-ALLOCATE ARRAY
    # ==========================================
    # Pre-allocate a raw NumPy array matching the exact shape of the DataFrame
    data_array = np.full((len(cluster_ids), len(columns)), np.nan)

    # Create fast lookup dictionaries to map keys to integer indices
    row_map = {cid: idx for idx, cid in enumerate(cluster_ids)}
    col_map = {col_tuple: idx for idx, col_tuple in enumerate(columns)}

    # ==========================================
    # 3. RUN THE LOOP & FILL THE ARRAY
    # ==========================================
    for cluster_id in tqdm(cluster_ids):
        loadname = bootstrap_dir / f'bootstrap_{cluster_id}.pkl'

        if not loadname.exists():
            raise ValueError(f'loadname does not exist: {loadname}')

        data: dict[str, PoissonOutput] = utils.load_obj(loadname)  # type: ignore

        cluster_x = cast(float, data_io.cluster_df.loc[cluster_id, 'cluster_x'])
        cluster_y = cast(float, data_io.cluster_df.loc[cluster_id, 'cluster_y'])

        # Get the row index for this cluster
        row_idx = row_map[cluster_id]

        for tid, tdata in data.items():
            # Skip if tid is not in our expected train_df (prevents KeyError)
            if tid not in data_io.train_df.index:
                continue

            if tdata is None:
                continue

            laser_x = cast(float, data_io.train_df.loc[tid, 'laser_x'])
            laser_y = cast(float, data_io.train_df.loc[tid, 'laser_y'])

            # Calculate Euclidean distance
            d = np.sqrt((cluster_x - laser_x) ** 2 + (cluster_y - laser_y) ** 2)

            # Write distance directly to the NumPy array
            col_idx = col_map[(tid, 'laser_distance')]
            data_array[row_idx, col_idx] = d

            # Write other metrics directly to the NumPy array
            for n in single_value_stats:
                col_idx = col_map[(tid, n)]
                val = tdata.get(n)
                # Ensure None values are safely written as np.nan
                data_array[row_idx, col_idx] = val if val is not None else np.nan

    # ==========================================
    # 4. POPULATE THE EMPTY DATAFRAME AT ONCE
    # ==========================================
    # Overwrite the empty DataFrame values with the completed NumPy array
    cell_responses.iloc[:, :] = data_array

    cell_responses.to_csv(savename)
    
    print(f'Saved: {savename}')


def analyse_responses(data_io: "DataIO", output_dir: Path, overwrite=False) -> None:
    """
    Handles calls to bootstrap function for single cells.
    """

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Detect per trial, which cell respond significantly
    num_threads: int = 20
    tasks: List[Dict[str, Any]] = []

    for cluster_id in data_io.cluster_df.index.values:
        savefile: Path = output_dir / f'bootstrap_{cluster_id}.pkl'
        if savefile.exists() and not overwrite:
            print('skipping')
            continue
        tasks.append({
            "data_io": data_io,
            "cluster_id": cluster_id,
            "savefile": savefile
        })

    if len(tasks) == 0:
        return

    utils.run_job(
        job_fn=bootstrap_data,
        tasks=tasks,
        num_threads=num_threads,
        debug=DEBUG,
    )


def bootstrap_data(
    data_io: "DataIO",
    cluster_id: str,
    savefile: Union[str, Path],
    debug_tid=None,
) -> None:
    """
    Perform bootstrap analysis for a single cluster and save results.

    Args:
        data_io: DataIO instance containing burst and spike information.
        cluster_id: ID of the cluster to process.
        savefile: Path to save the bootstrap results.
    """
    # print(f'Starting bootstrapping cluster: {cluster_id}')
    t_pre: int = 100
    t_after: int = 200
    stepsize: int = 10
    binwidth: int = 10
    bin_centres: np.ndarray = np.arange(-t_pre, t_after, stepsize)
    bin_alignment = 'left'
    baseline: List[int] = [-100, -50]
    response_window: List[int] = [0, 200]

    # min_inhibition_duration = 15  # minimum duration of inhibition in [ms]
    # min_excitation_duration = 15  # minimum duration of excitation in [ms]
    min_modulation_duration = 30

    # output_data: Dict[str, BootstrapOutput] = {}
    output_data: Dict[str, PoissonOutput] = {}

    for train_id in data_io.burst_df.train_id.unique():
        if debug_tid is not None and train_id != debug_tid:
            continue
        # results = BootstrapOutput()

        # Detect recording file of the current trial
        rec_id: str = str(data_io.train_df.loc[train_id, 'rec_id'])

        # Detect nr of bins
        n_bins: int = bin_centres.size

        # Find baseline index
        baseline_idx: np.ndarray = np.where((bin_centres >= baseline[0]) & (bin_centres <= baseline[1]))[0]

        # Detect burst onsets for this train
        has_dmd = data_io.train_df.loc[train_id, 'has_dmd']

        # If there is DMD stimulation; use dmd onset as 0, else use laser onset as 0
        if not has_dmd:
            burst_onsets: np.ndarray = data_io.burst_df.query(
                'train_id == @train_id').laser_burst_onset.values  # type: ignore
        else:
            burst_onsets: np.ndarray = data_io.burst_df.query('train_id == @train_id'
                                                  ).dmd_burst_onset.values  # type: ignore

        n_trains: int = len(burst_onsets)

        # Get spiketrain
        spiketrain: np.ndarray = data_io.spiketimes[rec_id][cluster_id]

        # # Create placeholder for data
        # binned_sp: np.ndarray = np.zeros((n_trains, n_bins), dtype=int)
        spike_times: List[np.ndarray] = []

        for burst_i, burst_onset in enumerate(burst_onsets):
            t0: float = burst_onset + bin_centres[0] - binwidth / 2
            t1: float = burst_onset + bin_centres[-1] + binwidth / 2
            idx: np.ndarray = np.where((spiketrain >= t0) & (spiketrain < t1))[0]
            # Append the spiketimes, relative to burst onset
            spike_times.append(spiketrain[idx] - burst_onset)

        # 1. Define the relative bin edges based on the alignment
        if bin_alignment == 'left':
            # [bin_centre, bin_centre + binwidth)
            bin_edges = np.append(bin_centres, bin_centres[-1] + binwidth)
        elif bin_alignment == 'centre':
            # [bin_centre - binwidth/2, bin_centre + binwidth/2)
            bin_edges = np.append(bin_centres - binwidth / 2, bin_centres[-1] + binwidth / 2)
        else:
            raise ValueError(f"Unknown bin alignment: {bin_alignment}")

        window_start = bin_edges[0]
        window_end = bin_edges[-1]

        # 2. Vectorized relative spike calculation
        # Broadcast subtraction: (n_trains, 1) - (1, n_spikes) -> (n_trains, n_spikes)
        # To save memory, we filter spikes that are globally near our window first
        global_min_onset = np.min(burst_onsets)
        global_max_onset = np.max(burst_onsets)

        # Filter the spiketrain to only include spikes that could fall into any window
        valid_spiketrain = spiketrain[
            (spiketrain >= global_min_onset + window_start) &
            (spiketrain <= global_max_onset + window_end)
            ]

        if valid_spiketrain.size < 10:
            output_data[train_id] = None
            continue

        # Calculate relative times (Shape: n_trains, n_valid_spikes)
        relative_spikes = valid_spiketrain[None, :] - burst_onsets[:, None]

        # Mask to find spikes that fall within our specific relative window
        valid_mask = (relative_spikes >= window_start) & (relative_spikes < window_end)

        # Get indices of the bursts (row indices) and the actual relative times
        burst_indices, spike_indices = np.where(valid_mask)
        valid_relative_times = relative_spikes[valid_mask]

        # 3. Fast 2D Histogram Binning (No loops!)
        binned_sp, _, _ = np.histogram2d(
            burst_indices,
            valid_relative_times,
            bins=[np.arange(n_trains + 1), bin_edges]
        )

        # Convert float counts to integers
        binned_sp = binned_sp.astype(int)

        # results.bins = bin_centres
        # results.bin_size = binwidth
        # results.binned_sp = binned_sp
        # results.spike_times = spike_times
        #
        # if np.sum(binned_sp) < 1 * 3:  # if there are less than 10 spikes don't bother
        #     results.reason = 'not enough spikes'
        #     continue

        # Bootstrap confidence intervals for each bin
        n_spikes_per_bin: np.ndarray = np.sum(binned_sp, axis=0)
        idx: np.ndarray = np.where(n_spikes_per_bin > 0)[0]

        result = detect_significant_modulation_poisson(
            bin_centres=bin_centres,
            binned_sp=binned_sp,
            baseline_idx=baseline_idx,
            min_duration_ms=min_modulation_duration,
            stepsize_ms=stepsize,
        )


        # try:
        #     btstrp = bootstrap(
        #         data=(binned_sp[:, idx],),
        #         vectorized=True,
        #         statistic=np.mean,
        #         axis=0,
        #         n_resamples=1000,
        #         confidence_level=0.95,
        #     )
        #     ci_low: np.ndarray = np.zeros(n_bins)
        #     ci_high: np.ndarray = np.zeros(n_bins)
        #     ci_low[idx] = btstrp.confidence_interval.low
        #     ci_high[idx] = btstrp.confidence_interval.high
        #     has_btrp: bool = True
        #
        # except ValueError:
        #     ci_low = np.zeros(n_bins)
        #     ci_high = np.zeros(n_bins)
        #     ci_low[idx] = np.nan
        #     ci_high[idx] = np.nan
        #     has_btrp = False

        # if not has_btrp:
        #     results.reason = 'bootstrap failed'
        #     continue
        #
        # ci_baseline: List[float] = [float(np.nanmean(ci_low[baseline_idx])), float(np.nanmean(ci_high[baseline_idx]))]
        #
        # # if ci_baseline[0] < 0.05:
        # #     ci_baseline[0] = 0
        #
        # results.firing_rate = np.mean(binned_sp, axis=0) / (binwidth / 1000)
        # results.firing_rate = medfilt(results.firing_rate, 3)
        # results.firing_rate_ci_high= ci_high / (binwidth / 1000)
        # results.firing_rate_ci_low= ci_low / (binwidth / 1000)
        #
        # # Detect which bins are increased relative to baseline
        # ex_idx = np.where((ci_low > ci_baseline[1]) & (bin_centres >= response_window[0]) &
        #                   (bin_centres < response_window[1]))[0]
        # ex_idx = first_consecutive_run(ex_idx, int(min_excitation_duration / stepsize))

        # Get statistics for excitatory response
        # results.excitation_bins = ex_idx
        # results.is_excited = True if ex_idx is not None else False
        # if results.is_excited:
        #     results.excitation_start = bin_centres[ex_idx[0]]
        #     results.excitation_end = bin_centres[ex_idx[-1]]
        #     results.excitation_duration = len(ex_idx) * stepsize
        #     results.excitation_max_fr = np.max(results.firing_rate[ex_idx])
        #
        # # Detect which bins are decrease relative to baseline
        # in_idx = np.where((ci_high < ci_baseline[0]) & (bin_centres >= response_window[0]) &
        #                   (bin_centres < response_window[1]))[0]
        # in_idx = first_consecutive_run(in_idx, int(min_inhibition_duration / stepsize))
        #
        # # Get statistics for excitatory response
        # results.inhibition_bins = in_idx
        # results.is_inhibited = True if in_idx is not None else False

        ###
        # if results.is_inhibited:
        #     results.inhibition_start = bin_centres[in_idx[0]]
        #     results.inhibition_end = bin_centres[in_idx[-1]]
        #     results.inhibition_duration = len(in_idx) * stepsize
        #     results.inhibition_min_fr = np.min(results.firing_rate[in_idx])

        output_data[train_id] = result

    utils.save_obj(output_data, savefile)  # type: ignore



def first_consecutive_run(indices, min_bin_length=3, occupancy_threshold=0.8):
    if len(indices) == 0:
        return None

    # Ensure indices are sorted for the sliding window
    indices = np.sort(indices)
    n = len(indices)

    # We use two pointers, i (start) and j (end)
    for i in range(n):
        for j in range(i + min_bin_length - 1, n):
            start = indices[i]
            end = indices[j]

            # The physical width (number of bins) this window covers
            span = end - start + 1

            # The number of indices we actually have inside this window
            actual_count = j - i + 1

            # We only care if the physical span is at least our minimum bin length
            if span >= min_bin_length:
                occupancy = actual_count / span

                if occupancy >= occupancy_threshold:
                    # Return the full continuous range from start to end
                    return np.arange(start, end + 1)

    return None



if __name__ == '__main__':
    main()
    # data_io = DataIO(datadir=dataset_dir)
    # data_io.load_session('2026-07-08 rat LE 3322 Mekano6 A')
    # cid = [c for c in data_io.cluster_ids if '_086' in c][0]
    # print(cid)
    # tid = [t for t in data_io.train_df.index.values if '_081' in t][0]
    # print(tid)
    #
    # bootstrap_data(
    #     data_io=data_io,
    #     cluster_id=cid,
    #     savefile='none',
    #     debug_tid=tid,
    # )

