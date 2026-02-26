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
from sonogenetics.analysis.analysis_params import dataset_dir
from data_io import DataIO


class BootstrapOutput:
    bins=None  # type: np.ndarray
    bin_size=None  # type: int
    binned_sp=None  # type: np.ndarray
    firing_rate=None  # type: np.ndarray
    firing_rate_ci_low=None  # type: np.ndarray
    firing_rate_ci_high=None  # type: np.ndarray
    spike_times=None  # type: List[np.ndarray]

    baseline_firing_rate=None

    is_excited=None
    excitation_bins=None
    excitation_max_fr=None
    excitation_start=None
    excitation_duration=None
    excitation_end=None

    is_inhibited=None
    inhibition_bins=None
    inhibition_min_fr=None
    inhibition_start=None
    inhibition_duration=None
    inhibition_end=None

    has_data=False
    reason='none'

    def get(self, name):
        assert hasattr(self, name), f'{name} not in attributes'
        return getattr(self, name)


DEBUG = True


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

    # Analyse the cell responses following the triggers
    analyse_responses(data_io, dataset_dir / 'bootstrapped')

    data_io.unlock_modification()

    # Gather all the response statistics into a single table
    gather_cluster_responses(data_io, dataset_dir / 'bootstrapped', dataset_dir / f'{data_io.session_id}_cells.csv')

    print('Done')


def gather_cluster_responses(data_io: DataIO, bootstrap_dir: Path, savename: Path):
    """
    gathers output from detect_significant_responses into a single dataframe`
    """

    # Names to store into cells dataframe
    names_to_register = [
        'is_excited',
        'is_inhibited',

        'excitation_max_fr',
        'excitation_start',
        'excitation_duration',

        'inhibition_min_fr',
        'inhibition_start',
        'inhibition_duration',

        'baseline_firing_rate',
        # 'laser_distance',

    ]

    # Setup pandas dataframe
    columns: List[Tuple[str, str]] = []
    for train_id in data_io.train_df.index.values:
        for n in names_to_register:
            columns.append((train_id,n))
    multi_index = pd.MultiIndex.from_tuples(columns)

    cell_responses = pd.DataFrame(index=data_io.cluster_df.index.values, columns=multi_index)

    for cluster_id in tqdm(cell_responses.index.values):
        loadname = bootstrap_dir / f'bootstrap_{cluster_id}.pkl'

        if not loadname.exists():
            raise ValueError(f'loadname does not exist: {loadname}')

        data: dict[str, BootstrapOutput] = utils.load_obj(loadname)  # type: ignore

        cluster_x = cast(float, data_io.cluster_df.loc[cluster_id, 'cluster_x'])
        cluster_y = cast(float, data_io.cluster_df.loc[cluster_id, 'cluster_y'])


        for tid, tdata in data.items():
            laser_x: float = cast(float, data_io.train_df.loc[tid, 'laser_x'])
            laser_y: float = cast(float, data_io.train_df.loc[tid, 'laser_y'])

            d = np.sqrt((cluster_x - laser_x)**2 + (cluster_y - laser_y)**2)
            cell_responses.at[cluster_id, (tid, 'laser_distance')] = d

            for n in names_to_register:
                cell_responses.at[cluster_id, (tid, n)] = data[tid].get(n)

    cell_responses.to_csv(savename)
    
    print(f'Saved: {savename}')


def analyse_responses(data_io: "DataIO", output_dir: Path) -> None:
    """
    Handles calls to bootstrap function for single cells.
    """

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Detect per trial, which cell respond significantly
    num_threads: int = 10
    tasks: List[Dict[str, Any]] = []

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

    utils.run_job(
        job_fn=bootstrap_data,
        tasks=tasks,
        num_threads=num_threads,
        debug=DEBUG,
    )


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

    min_inhibition_duration = 15  # minimum duration of inhibition in [ms]
    min_excitation_duration = 15  # minimum duration of excitation in [ms]

    output_data: Dict[str, BootstrapOutput] = {}

    for train_id in data_io.burst_df.train_id.unique():
        results = BootstrapOutput()

        # Detect recording file of the current trial
        rec_id: str = str(data_io.train_df.loc[train_id, 'rec_id'])

        # Detect nr of bins
        n_bins: int = bin_centres.size

        # Find baseline index
        baseline_idx: np.ndarray = np.where((bin_centres >= baseline[0]) & (bin_centres <= baseline[1]))[0]

        # Detect burst onsets for this train
        if data_io.train_df.loc[train_id, 'has_laser']:
            burst_onsets: np.ndarray = data_io.burst_df.query(
                'train_id == @train_id').laser_burst_onset.values  # type: ignore
        else:
            burst_onsets = data_io.burst_df.query('train_id == @train_id'
                                                  ).dmd_burst_onset.values  # type: ignore

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

        results.bins = bin_centres
        results.bin_size = binwidth
        results.binned_sp = binned_sp
        results.spike_times = spike_times

        if np.sum(binned_sp) < 1 * 3:  # if there are less than 10 spikes don't bother
            results.reason = 'not enough spikes'
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
            results.reason = 'bootstrap failed'
            continue

        ci_baseline: List[float] = [float(np.nanmean(ci_low[baseline_idx])), float(np.nanmean(ci_high[baseline_idx]))]

        if ci_baseline[0] < 0.05:
            ci_baseline[0] = 0

        results.firing_rate = np.mean(binned_sp, axis=0) / (binwidth / 1000)
        results.firing_rate = medfilt(results.firing_rate, 3)
        results.firing_rate_ci_high= ci_high / (binwidth / 1000)
        results.firing_rate_ci_low= ci_low / (binwidth / 1000)

        # Detect which bins are increased relative to baseline
        ex_idx = np.where((ci_low > ci_baseline[1]) & (bin_centres >= response_window[0]) &
                          (bin_centres < response_window[1]))[0]
        ex_idx = first_consecutive_run(ex_idx, int(min_excitation_duration / stepsize))

        # Get statistics for excitatory response
        results.excitation_bins = ex_idx
        results.is_excited = True if ex_idx is not None else False
        if results.is_excited:
            results.excitation_start = bin_centres[ex_idx[0]]
            results.excitation_end = bin_centres[ex_idx[-1]]
            results.excitation_duration = len(ex_idx) * stepsize
            results.excitation_max_fr = np.max(results.firing_rate[ex_idx])

        # Detect which bins are decrease relative to baseline
        in_idx = np.where((ci_high < ci_baseline[0]) & (bin_centres >= response_window[0]) &
                          (bin_centres < response_window[1]))[0]
        in_idx = first_consecutive_run(in_idx, int(min_inhibition_duration / stepsize))

        # Get statistics for excitatory response
        results.inhibition_bins = in_idx
        results.is_inhibited = True if in_idx is not None else False
        if results.is_inhibited:
            results.inhibition_start = bin_centres[in_idx[0]]
            results.inhibition_end = bin_centres[in_idx[-1]]
            results.inhibition_duration = len(in_idx) * stepsize
            results.inhibition_min_fr = np.min(results.firing_rate[in_idx])

        output_data[train_id] = results

    utils.save_obj(output_data, savefile)  # type: ignore


def first_consecutive_run(indices, min_bin_length=3):
    if len(indices) == 0:
        return None

    run_start = indices[0]
    run_length = 1

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            run_length += 1
        else:
            if run_length > min_bin_length:
                return np.arange(run_start, run_start + run_length)
            run_start = indices[i]
            run_length = 1

    # Check final run
    if run_length > min_bin_length:
        return np.arange(run_start, run_start + run_length)

    return None



if __name__ == '__main__':
    main()

