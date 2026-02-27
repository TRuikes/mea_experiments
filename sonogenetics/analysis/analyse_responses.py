import sys
from pathlib import Path
sys.path.append('.')
current_dir = Path().resolve()
sys.path.append(current_dir.as_posix().split('mea_experiments')[0] + 'mea_experiments')

import pandas as pd
import threading
from tqdm import tqdm
import numpy as np
from scipy.stats import bootstrap
from scipy.signal import medfilt
from typing import List, Tuple, cast, Optional, Any, Dict, Union
from pathlib import Path

import utils
from sonogenetics.analysis.analysis_params import dataset_dir
from sonogenetics.preprocessing.dataset_sessions import dataset_sessions
from data_io import DataIO

DEBUG = False

class BootstrapOutput:
    is_excited=None
    is_inhibited=None
    excitation_max_fr=None
    excitation_start=None
    excitation_duration=None
    inhibition_min_fr=None
    inhibition_start=None
    inhibition_duration=None
    baseline_firing_rate=None
    laser_distance=None


def main():
    """
    Main handles
    """
    data_io = DataIO(dataset_dir)
    session_id = '2026-02-19 mouse c57 5713 Mekano6 A'

    # session_id = data_io.sessions[0]
    print(f'Loading data: {session_id}')
    data_io.load_session(session_id, load_waveforms=False, load_pickle=False )  # type: ignore
    data_io.dump_as_pickle()
    data_io.lock_modification()
    detect_significant_responses(data_io, dataset_dir / 'bootstrapped')

    print(f'Gathering results')
    gather_cluster_responses(data_io, dataset_dir / 'bootstrapped', dataset_dir / f'{data_io.session_id}_cells.csv')

    print('Done')
    data_io.unlock_modification()


def gather_cluster_responses(data_io: DataIO, bootstrap_dir: Path, savename: Path):
    """
    gathers output from detect_significant_responses into a single dataframe`
    """
    baseline_t0 = -200
    baseline_t1 = -100

    names_to_register = [
        'is_significant',
        'response_firing_rate',
        'baseline_firing_rate',
        'response_latency',
        'response_duration',
        'response_type',
        'laser_distance',
    ]

    columns: List[Tuple[str, str]] = []
    for burst_id in data_io.burst_df.train_id.unique():
        for n in names_to_register:
            columns.append((burst_id,n))
    multi_index = pd.MultiIndex.from_tuples(columns)

    cell_responses = pd.DataFrame(index=data_io.cluster_df.index.values, columns=multi_index)


    for cluster_id in tqdm(cell_responses.index.values):
        loadname = bootstrap_dir / f'bootstrap_{cluster_id}.pkl'

        if not loadname.exists():
            raise ValueError(f'loadname does not exist: {loadname}')

        data: dict[str, dict[str, bool]] = utils.load_obj(loadname)  # type: ignore

        cluster_x = cast(float, data_io.cluster_df.loc[cluster_id, 'cluster_x'])
        cluster_y = cast(float, data_io.cluster_df.loc[cluster_id, 'cluster_y'])


        for tid, tdata in data.items():
            row_index = data_io.burst_df.index[data_io.burst_df['train_id'] == tid ]
            laser_x: float = cast(float, data_io.burst_df.loc[row_index, 'laser_x'])
            laser_y: float = cast(float, data_io.burst_df.loc[row_index, 'laser_y'])

            d = np.sqrt((cluster_x - laser_x)**2 + (cluster_y - laser_y)**2)
            cell_responses.at[cluster_id, (tid, 'laser_distance')] = d
            cell_responses.at[cluster_id, (tid, 'is_significant')] = tdata['is_sig']

            # Detect indices for baseline and response times
            bin_centres: np.ndarray = tdata['bins']  # type: ignore
            baseline_idx = np.where((bin_centres >= baseline_t0) & (bin_centres <= baseline_t1))[0]
            response_idx = np.where((bin_centres > 0) & (bin_centres <= 200))[0]

            # smooth the firing rate a little
            firing_rate: np.ndarray = tdata['firing_rate']  # type: ignore

            if firing_rate is None:  # type: ignore
                continue

            firing_rate_smooth = medfilt(firing_rate, 3)

            # Detect if the cell is inhibited or excited
            mean_baseline_fr = np.nanmean(firing_rate_smooth[baseline_idx])
            mean_response_fr = np.nanmean(firing_rate_smooth[response_idx])

            if mean_response_fr < mean_baseline_fr:
                response_type = 'inhibited'
            else:
                response_type = 'excited'

            cell_responses.at[cluster_id, (tid, 'response_type')] = response_type
            cell_responses.at[cluster_id, (tid, 'baseline_firing_rate')] = mean_baseline_fr

            # Detect minimum or maximum firing rate
            if response_type == 'inhibited':
                cell_responses.at[cluster_id, (tid, 'response_firing_rate')] = np.nanmin(
                    firing_rate_smooth[response_idx])
            else:
                cell_responses.at[cluster_id, (tid, 'response_firing_rate')] = np.nanmax(
                    firing_rate_smooth[response_idx])

            if not tdata['is_sig']:
                continue

            sig_idx: np.ndarray = tdata['significant_bins']  # type: ignore

            # Find the first consecutive sentence of significant bins, following
            # burst onset
            b0 = np.where(bin_centres >= 0)[0][0]
            s = [s for s in sig_idx if s >= b0]
            res = find_first_long_consecutive_sequence(s, min_length=3)

            if res is None:
                latency = None
            else:
                latency = bin_centres[res[0]]

            cell_responses.at[cluster_id, (tid, 'response_latency')] = latency

    cell_responses.to_csv(savename)
    
    print(f'Saved: {savename}')



def find_first_long_consecutive_sequence(
    nums: List[int], 
    min_length: int = 3
) -> Optional[List[int]]:
    """
    Find the first sequence of consecutive integers in `nums` longer than `min_length`.
    Returns the sequence as a list, or None if no such sequence exists.
    """
    start: Optional[int] = None
    length: int = len(nums)

    for i in range(length - 1):
        if nums[i + 1] == nums[i] + 1:
            if start is None:
                start = i
        else:
            if start is not None:
                if (i - start + 1) > min_length:
                    # Return the first sequence of consecutive integers longer than min_length
                    return nums[start:i + 1]
            start = None

    # Check for the case where the sequence ends at the last element
    if start is not None and (length - start) > min_length:
        return nums[start:length]

    return None



def detect_significant_responses(data_io: "DataIO", output_dir: Path) -> None:
    """
    Handles calls to bootstrap function for single cells.
    """

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Detect per trial, which cell respond significantly
    num_threads: int = 10
    threads: List[threading.Thread] = []
    tasks: List[Dict[str, Any]] = []
    lock: threading.Lock = threading.Lock()

    for cluster_id in data_io.cluster_df.index.values:
        savefile: Path = output_dir / f'bootstrap_{cluster_id}.pkl'
        if savefile.exists():
            continue
        tasks.append({
            "data_io": data_io,
            "cluster_id": cluster_id,
            "savefile": savefile
        })

    if len(tasks) == 0:
        return

    # Check if we are in debugging mode
    if not DEBUG:
        with tqdm(total=len(tasks)) as progress_bar:
            for _ in range(num_threads):
                t: threading.Thread = threading.Thread(target=thread_task,  # type: ignore
                                                    args=(tasks, progress_bar, lock))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

    else:
        n_tasks = len(tasks)
        for i in range(n_tasks):
            bootstrap_data(**tasks[i])
            print(f'Completed {i+1}/{n_tasks}')



def thread_task(
    tasks: List[Dict[str, Any]],
    progress_bar: Optional[tqdm],  # type: ignore
    lock: threading.Lock
) -> None:
    """
    Worker thread for processing bootstrap tasks.

    Args:
        tasks: List of dictionaries containing task arguments.
        progress_bar: tqdm progress bar instance (or None).
        lock: threading.Lock to synchronize access to tasks and progress bar.
    """
    while tasks:
        with lock:
            if not tasks:
                break

        task = tasks.pop(0)
        bootstrap_data(**task)

        with lock:
            if progress_bar is not None:
                progress_bar.update(1)



def bootstrap_data(
    data_io: "DataIO",
    cluster_id: str,
    savefile: Union[str, Path]
) -> None:
    """
    Perform bootstrap analysis for a single cluster and save results.

    Args:
        data_io: DataIO instance containing burst and spike information.
        cluster_id: ID of the cluster to process.
        savefile: Path to save the bootstrap results.
    """
    print(f'Starting bootstrapping cluster: {cluster_id}')
    t_pre: int = 200
    t_after: int = 500
    stepsize: int = 5
    binwidth: int = 20
    bin_centres: np.ndarray = np.arange(-t_pre, t_after, stepsize)
    baseline: List[int] = [-200, -100]
    response_window: List[int] = [0, 200]

    output_data: Dict[Union[int, str], Dict[str, Any]] = {}

    for train_id in data_io.burst_df.train_id.unique():
        output_data[train_id] = dict(
            bins=None,  # type: np.ndarray
            bin_size=None,  # type: int
            binned_sp=None,  # type: np.ndarray
            firing_rate=None,  # type: np.ndarray
            firing_rate_ci_low=None,  # type: np.ndarray
            firing_rate_ci_high=None,  # type: np.ndarray
            spike_times=None,  # type: List[np.ndarray]
            significant_bins=None,  # type: List[int]
            is_sig=None,  # type: bool
            has_data=False,
            reason='none',
        )

        # Detect recording file of the current trial
        rec_id: str = str(data_io.burst_df.query('train_id == @train_id').iloc[0].rec_id)  # type: ignore

        # Detect nr of bins
        n_bins: int = bin_centres.size

        # Find baseline index
        baseline_idx: np.ndarray = np.where((bin_centres >= baseline[0]) & (bin_centres <= baseline[1]))[0]

        # Find response window index
        response_idx: np.ndarray = np.where((bin_centres >= response_window[0]) & (bin_centres <= response_window[1]))[0]

        # Detect burst onsets for this train
        if 'has_laser' in data_io.burst_df.columns:
            if data_io.burst_df.query('train_id == @train_id').iloc[0].has_laser:
                stimtype = 'laser'
            else:
                stimtype = 'dmd'
        else:
            stimtype: str = str(data_io.burst_df.query('train_id == @train_id').iloc[0].stimtype)
            # type: ignore
        if stimtype in ('laser', 'padmd'):
            burst_onsets: np.ndarray = data_io.burst_df.query('train_id == @train_id').laser_burst_onset.values # type: ignore
        elif stimtype == 'dmd':
            burst_onsets = data_io.burst_df.query('train_id == @train_id').dmd_burst_onset.values # type: ignore
        else:
            burst_onsets = np.array([])

        n_trains: int = len(burst_onsets)

        # Get spiketrain
        spiketrain: np.ndarray = data_io.spiketimes[rec_id][cluster_id]

        # Create placeholder for data
        binned_sp: np.ndarray = np.zeros((n_trains, n_bins), dtype=int)
        spike_times: List[np.ndarray] = []

        for burst_i, burst_onset in enumerate(burst_onsets):
            t0: float = burst_onset + bin_centres[0] - binwidth / 2
            t1: float = burst_onset + bin_centres[-1] + binwidth / 2
            idx: np.ndarray = np.where((spiketrain >= t0) & (spiketrain < t1))[0]

            # Append the spiketimes, relative to burst onset
            spike_times.append(spiketrain[idx] - burst_onset)

            for bin_i, bin_centre in enumerate(bin_centres):
                t0 = burst_onset + bin_centre - binwidth / 2
                t1 = burst_onset + bin_centre + binwidth / 2
                idx = np.where((spiketrain >= t0) & (spiketrain < t1))[0]
                binned_sp[burst_i, bin_i] = idx.size

        output_data[train_id]['bins'] = bin_centres
        output_data[train_id]['bin_size'] = binwidth
        output_data[train_id]['binned_sp'] = binned_sp
        output_data[train_id]['spike_times'] = spike_times

        if np.sum(binned_sp) < 1 * 3:  # if there are less than 10 spikes don't bother
            output_data[train_id]['reason'] = 'not enough spikes'
            continue

        # Bootstrap confidence intervals for each bin
        n_spikes_per_bin: np.ndarray = np.sum(binned_sp, axis=0)
        idx: np.ndarray = np.where(n_spikes_per_bin > 0)[0]

        try:
            btstrp = bootstrap(
                data=(binned_sp[:, idx],),
                vectorized=True,
                statistic=np.mean,
                axis=0,
                n_resamples=1000,
                confidence_level=0.95,
            )
            ci_low: np.ndarray = np.zeros(n_bins)
            ci_high: np.ndarray = np.zeros(n_bins)
            ci_low[idx] = btstrp.confidence_interval.low
            ci_high[idx] = btstrp.confidence_interval.high
            has_btrp: bool = True

        except ValueError:
            ci_low = np.zeros(n_bins)
            ci_high = np.zeros(n_bins)
            ci_low[idx] = np.nan
            ci_high[idx] = np.nan
            has_btrp = False

        if not has_btrp:
            output_data[train_id]['reason'] = 'bootstrap failed'
            continue

        ci_baseline: List[float] = [float(np.nanmean(ci_low[baseline_idx])), float(np.nanmean(ci_high[baseline_idx]))]

        if ci_baseline[0] < 0.05:
            ci_baseline[0] = 0

        # Detect which bins do not overlap with baseline bins
        sig_idx: List[int] = [i for i in response_idx if not intervals_overlap((ci_low[i], ci_high[i]), ci_baseline)]  # type: ignore

        output_data[train_id]['is_sig'] = True if len(sig_idx) * stepsize >= 25 else False
        output_data[train_id]['significant_bins'] = sig_idx

        firing_rate: np.ndarray = np.mean(binned_sp, axis=0) / (binwidth / 1000)
        firing_rate_ci_high: np.ndarray = ci_high / (binwidth / 1000)
        firing_rate_ci_low: np.ndarray = ci_low / (binwidth / 1000)

        output_data[train_id]['firing_rate'] = firing_rate
        output_data[train_id]['firing_rate_ci_low'] = firing_rate_ci_low
        output_data[train_id]['firing_rate_ci_high'] = firing_rate_ci_high

    utils.save_obj(output_data, savefile)  # type: ignore


def intervals_overlap(
    interval1: Tuple[Union[int, float], Union[int, float]],
    interval2: Tuple[Union[int, float], Union[int, float]]
) -> bool:
    """
    Checks if two intervals overlap.

    Args:
        interval1: Tuple of (start, end) for the first interval.
        interval2: Tuple of (start, end) for the second interval.

    Returns:
        True if intervals overlap, False otherwise.
    """
    a, b = interval1
    c, d = interval2
    return max(a, c) <= min(b, d)


if __name__ == '__main__':
    main()

