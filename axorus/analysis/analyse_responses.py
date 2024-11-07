from axorus.data_io import DataIO
from pathlib import Path
import pandas as pd
import threading
import time
from tqdm import tqdm
import numpy as np
from scipy.stats import bootstrap
import utils
from scipy.signal import medfilt
import sys


def main():
    """
    Main handles
    """
    dataset_dir = Path(r'D:\Axorus\ex_vivo_series_3\dataset')
    assert dataset_dir.exists(), f'cant find: {dataset_dir}'
    data_io = DataIO(dataset_dir)

    data_io.load_session(data_io.sessions[0])
    data_io.lock_modification()
    detect_significant_responses(data_io, dataset_dir / 'bootstrapped')
    gather_cluster_responses(data_io, dataset_dir / 'bootstrapped', dataset_dir / f'{data_io.session_id}_cells.csv')
    data_io.unlock_modification()


def gather_cluster_responses(data_io: DataIO, bootstrap_dir: Path, savename: Path):
    """
    gathers output from detect_significant_responses into a single dataframe
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
    ]
    columns = []
    for burst_id in data_io.burst_df.train_id.unique():
        for n in names_to_register:
            columns.append((burst_id,n))
    multi_index = pd.MultiIndex.from_tuples(columns)

    cell_responses = pd.DataFrame(index=data_io.cluster_df.index.values, columns=multi_index)

    for cluster_id in cell_responses.index.values:
        loadname = bootstrap_dir / f'bootstrap_{cluster_id}.pkl'

        if not loadname.exists():
            raise ValueError(f'loadname does not exist: {loadname}')

        data = utils.load_obj(loadname)

        for tid, tdata in data.items():
            cell_responses.at[cluster_id, (tid, 'is_significant')] = tdata['is_sig']

            # Detect indices for baseline and response times
            bin_centres = tdata['bins']
            baseline_idx = np.where((bin_centres >= baseline_t0) & (bin_centres <= baseline_t1))[0]
            response_idx = np.where((bin_centres > 0) & (bin_centres <= 200))[0]

            # smooth the firing rate a little
            firing_rate = tdata['firing_rate']

            if firing_rate is None:
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

            sig_idx = tdata['significant_bins']

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


def find_first_long_consecutive_sequence(nums, min_length=3):
    """
    thanks to chatGPT; does what the function name says!
    """
    start = None
    length = len(nums)

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


def detect_significant_responses(data_io: DataIO, output_dir: Path):
    """
    Handles calls to bootstrap function for single cells
    """

    # Detect per trial, which cell respond significantly
    num_threads = 5
    threads = []
    tasks = []
    lock = threading.Lock()

    for cluster_id in data_io.cluster_df.index.values:
        savefile = output_dir / f'bootstrap_{cluster_id}.pkl'
        if savefile.exists():
            continue
        tasks.append(dict(data_io=data_io, cluster_id=cluster_id,
                          savefile=savefile))

    if len(tasks) == 0:
        return

    # Check if we are in debugging mode
    with tqdm(total=len(tasks)) as progress_bar:
        for _ in range(num_threads):
            t = threading.Thread(target=thread_task, args=(tasks, progress_bar, lock))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    # UNCOMMENT THIS TO DEBUG
    # n_tasks = len(tasks)
    # for i in range(n_tasks):
    #     thread_task(tasks, None, lock)


def thread_task(tasks, progress_bar, lock):
    """

    :param tasks:
    :param progress_bar:
    :param cell_responses:
    :param lock:
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


# noinspection PyTypedDict
def bootstrap_data(data_io: DataIO, cluster_id: str, savefile: str):
    """

    :param data_io:
    :param cluster_id:
    :param savefile:
    """
    t_pre = 200
    t_after = 500
    stepsize = 5
    binwidth = 20
    bin_centres = np.arange(-t_pre, t_after, stepsize)
    baseline = [-200, -100]

    output_data = {}
    for train_id in data_io.burst_df.train_id.unique():
        output_data[train_id] = dict(
            bins=None,
            bin_size=None,
            binned_sp=None,
            firing_rate=None,
            firing_rate_ci_low=None,
            firing_rate_ci_high=None,
            spike_times=None,
            significant_bins=None,
            is_sig=None,
            has_data=False,
            reason='none',
        )

        # Detect recording file of the current trial
        rec_id = data_io.burst_df.query('train_id == @train_id').iloc[0].rec_id

        # Detect nr of bins
        n_bins = bin_centres.size

        # Find baseline index
        baseline_idx = np.where((bin_centres >= baseline[0]) & (bin_centres <= baseline[1]))[0]

        # Detect burst onsets for this train
        burst_onsets = data_io.burst_df.query('train_id == @train_id').burst_onset.values
        n_trains = len(burst_onsets)

        # Get spiketrain
        spiketrain = data_io.spiketimes[rec_id][cluster_id]

        # Create placeholder for data
        binned_sp = np.zeros((n_trains, n_bins), dtype=int)
        spike_times = []

        for burst_i, burst_onset in enumerate(burst_onsets):
            t0 = burst_onset + bin_centres[0] - binwidth / 2
            t1 = burst_onset + bin_centres[-1] + binwidth / 2
            idx = np.where((spiketrain >= t0) & (spiketrain < t1))[0]

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
        n_spikes_per_bin = np.sum(binned_sp, axis=0)
        idx = np.where(n_spikes_per_bin > 0)[0]

        try:
            btstrp = bootstrap(
                data=(binned_sp[:, idx],),
                vectorized=True,
                statistic=np.mean,
                axis=0,
                n_resamples=1000,
                confidence_level=0.95,
            )
            ci_low, ci_high = np.zeros(n_bins), np.zeros(n_bins)
            ci_low[idx] = btstrp.confidence_interval.low
            ci_high[idx] = btstrp.confidence_interval.high
            has_btrp = True

        except ValueError:
            ci_low, ci_high = np.zeros(n_bins), np.zeros(n_bins)
            ci_low[idx] = np.nan
            ci_high[idx] = np.nan
            has_btrp = False

        if not has_btrp:
            output_data[train_id]['reason'] = 'bootstrap failed'
            continue

        ci_baseline = [np.nanmean(ci_low[baseline_idx]), np.nanmean(ci_high[baseline_idx])]

        # If confidence interval of baseline is close to 0, set to 0
        if ci_baseline[0] < 0.05:
            ci_baseline[0] = 0

        # Detect which bins do not overlap with baseline bins
        sig_idx = [i for i in range(n_bins) if not intervals_overlap((ci_low[i], ci_high[i]), ci_baseline)]

        # Detect stepsize of the bins
        stepsize = np.diff(bin_centres)[0]

        # Determine if the response was truly significant
        output_data[train_id]['is_sig'] = True if len(sig_idx) * stepsize >= 25 else False
        output_data[train_id]['significant_bins'] = sig_idx

        firing_rate = np.mean(binned_sp, axis=0) / (binwidth / 1000)
        firing_rate_ci_high = ci_high / (binwidth / 1000)
        firing_rate_ci_low = ci_low / (binwidth / 1000)

        output_data[train_id]['firing_rate'] = firing_rate
        output_data[train_id]['firing_rate_ci_low'] = firing_rate_ci_low
        output_data[train_id]['firing_rate_ci_high'] = firing_rate_ci_high

    utils.save_obj(output_data, savefile)


def intervals_overlap(interval1, interval2):
    """"
    thanks to chatGPT; checks if two intervals overlap
    """
    a, b = interval1
    c, d = interval2
    return max(a, c) <= min(b, d)


if __name__ == '__main__':
    main()

