import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import to_rgb
import seaborn as sns
import os
import pickle

from psth_params import *


# ============================================================
# STATISTICAL TESTING FUNCTIONS
# ============================================================

def bootstrap_test(baseline_data, response_data, n_iterations=1000, test_type='excitatory'):
    """
    Bootstrap test for significant response.
    
    Parameters
    ----------
    baseline_data : array
        Firing rates during baseline period
    response_data : array
        Firing rates during response window
    n_iterations : int
        Number of bootstrap iterations
    test_type : str
        'excitatory' or 'inhibitory'
    
    Returns
    -------
    p_value : float
        Bootstrap p-value
    """
    if len(baseline_data) == 0 or len(response_data) == 0:
        return 1.0
    
    # Observed difference
    obs_diff = np.mean(response_data) - np.mean(baseline_data)
    
    # Bootstrap distribution under null hypothesis
    combined = np.concatenate([baseline_data, response_data])
    n_baseline = len(baseline_data)
    n_response = len(response_data)
    
    bootstrap_diffs = []
    for _ in range(n_iterations):
        # Resample with replacement
        resampled = np.random.choice(combined, size=len(combined), replace=True)
        boot_baseline = resampled[:n_baseline]
        boot_response = resampled[n_baseline:]
        boot_diff = np.mean(boot_response) - np.mean(boot_baseline)
        bootstrap_diffs.append(boot_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Calculate p-value
    if test_type == 'excitatory':
        # One-tailed: response > baseline
        p_value = np.mean(bootstrap_diffs >= obs_diff)
    else:  # inhibitory
        # One-tailed: response < baseline
        p_value = np.mean(bootstrap_diffs <= obs_diff)
    
    return p_value

def welch_t_test(baseline_data, response_data, test_type='excitatory'):
    """
    Welch's t-test for unequal variances.
    
    Parameters
    ----------
    baseline_data : array
        Firing rates during baseline period
    response_data : array
        Firing rates during response window
    test_type : str
        'excitatory' or 'inhibitory'
    
    Returns
    -------
    p_value : float
        t-test p-value
    """
    if len(baseline_data) < 2 or len(response_data) < 2:
        return 1.0
    
    if test_type == 'excitatory':
        # One-tailed: response > baseline
        t_stat, p_value = stats.ttest_ind(response_data, baseline_data, 
                                          equal_var=False, alternative='greater')
    else:  # inhibitory
        # One-tailed: response < baseline
        t_stat, p_value = stats.ttest_ind(response_data, baseline_data, 
                                          equal_var=False, alternative='less')
    
    return p_value

def permutation_test(baseline_data, response_data, n_permutations=1000, test_type='excitatory'):
    """
    Permutation test for significant response.
    
    Parameters
    ----------
    baseline_data : array
        Firing rates during baseline period
    response_data : array
        Firing rates during response window
    n_permutations : int
        Number of permutations
    test_type : str
        'excitatory' or 'inhibitory'
    
    Returns
    -------
    p_value : float
        Permutation test p-value
    """
    if len(baseline_data) == 0 or len(response_data) == 0:
        return 1.0
    
    # Observed difference
    obs_diff = np.mean(response_data) - np.mean(baseline_data)
    
    # Combine data
    combined = np.concatenate([baseline_data, response_data])
    n_baseline = len(baseline_data)
    
    # Permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        # Randomly shuffle and split
        np.random.shuffle(combined)
        perm_baseline = combined[:n_baseline]
        perm_response = combined[n_baseline:]
        perm_diff = np.mean(perm_response) - np.mean(perm_baseline)
        perm_diffs.append(perm_diff)
    
    perm_diffs = np.array(perm_diffs)
    
    # Calculate p-value
    if test_type == 'excitatory':
        p_value = np.mean(perm_diffs >= obs_diff)
    else:
        p_value = np.mean(perm_diffs <= obs_diff)
    
    return p_value

def sliding_window_test(rate_smooth, baseline_mean, baseline_std, test_type='excitatory'):
    """
    Sliding window t-test to find significant response windows.
    
    Parameters
    ----------
    rate_smooth : array
        Smoothed firing rate over time
    baseline_mean : float
        Baseline mean firing rate
    baseline_std : float
        Baseline standard deviation
    test_type : str
        'excitatory' or 'inhibitory'
    
    Returns
    -------
    min_p_value : float
        Minimum p-value across all windows
    best_latency_ms : float
        Latency of window with minimum p-value
    """
    window_bins = int(np.round(WINDOW_SIZE_MS / BIN_SIZE_MS))
    step_bins = int(np.round(WINDOW_STEP_MS / BIN_SIZE_MS))
    
    # Search range
    search_mask = (t_centers >= MIN_LATENCY_MS) & (t_centers <= POST_TIME_MS)
    search_indices = np.where(search_mask)[0]
    
    if len(search_indices) < window_bins:
        return 1.0, np.nan
    
    min_p = 1.0
    best_latency = np.nan
    
    # Slide window
    for start_idx in range(search_indices[0], search_indices[-1] - window_bins, step_bins):
        window_data = rate_smooth[start_idx:start_idx + window_bins]
        
        if len(window_data) < 2:
            continue
        
        # One-sample t-test against baseline
        if test_type == 'excitatory':
            t_stat, p_val = stats.ttest_1samp(window_data, baseline_mean, 
                                              alternative='greater')
        else:
            t_stat, p_val = stats.ttest_1samp(window_data, baseline_mean, 
                                              alternative='less')
        
        if p_val < min_p:
            min_p = p_val
            best_latency = t_centers[start_idx]
    
    return min_p, best_latency

def test_response_significance(rate_smooth, spike_counts_baseline, spike_counts_response, 
                               mu, sd, test_type='excitatory'):
    """
    Run all enabled statistical tests and combine results.
    
    Parameters
    ----------
    rate_smooth : array
        Smoothed firing rate
    spike_counts_baseline : array
        Spike counts per bin during baseline
    spike_counts_response : array
        Spike counts per bin during response window
    mu : float
        Baseline mean
    sd : float
        Baseline standard deviation
    test_type : str
        'excitatory' or 'inhibitory'
    
    Returns
    -------
    dict with p-values from each test and combined significance
    """
    if not ENABLE_STATISTICAL_TESTS:
        return {
            'significant': True,
            'combined_p': 0.0,
            'test_results': {}
        }
    
    results = {}
    p_values = []
    
    # Bootstrap test
    if STATISTICAL_TESTS['bootstrap']:
        p_boot = bootstrap_test(spike_counts_baseline, spike_counts_response, 
                               N_BOOTSTRAP, test_type)
        results['bootstrap'] = p_boot
        p_values.append(p_boot)
    
    # Welch's t-test
    if STATISTICAL_TESTS['t_test']:
        p_ttest = welch_t_test(spike_counts_baseline, spike_counts_response, test_type)
        results['t_test'] = p_ttest
        p_values.append(p_ttest)
    
    # Permutation test
    if STATISTICAL_TESTS['permutation']:
        p_perm = permutation_test(spike_counts_baseline, spike_counts_response, 
                                 N_PERMUTATIONS, test_type)
        results['permutation'] = p_perm
        p_values.append(p_perm)
    
    # Sliding window test
    if STATISTICAL_TESTS['sliding_window']:
        p_slide, slide_lat = sliding_window_test(rate_smooth, mu, sd, test_type)
        results['sliding_window'] = p_slide
        results['sliding_window_latency'] = slide_lat
        p_values.append(p_slide)
    
    # Combine p-values (use minimum p-value with Bonferroni correction if enabled)
    if len(p_values) > 0:
        combined_p = np.min(p_values)
        if BONFERRONI_CORRECTION:
            alpha_corrected = SIGNIFICANCE_ALPHA / len(p_values)
        else:
            alpha_corrected = SIGNIFICANCE_ALPHA
        
        significant = combined_p < alpha_corrected
    else:
        combined_p = 1.0
        significant = True  # No tests enabled, accept response
        alpha_corrected = SIGNIFICANCE_ALPHA
    
    return {
        'significant': significant,
        'combined_p': combined_p,
        'alpha_corrected': alpha_corrected,
        'test_results': results
    }

# ============================================================
# COLOR UTIL
# ============================================================

def color_dc(dc):
    match dc:
        case 8:
            shade = 3
        case 12:
            shade = 2
        case 20:
            shade = 1
        case 29:
            shade = 0

    palette = sns.color_palette('inferno', 4)
    return palette[shade]


# ============================================================
# LATENCY DETECTION WITH STATISTICAL TESTING
# ============================================================

def detect_latency(rate_smooth, mu, sd, spike_counts_all, spike_counts_baseline):
    """
    Detect latency with statistical testing.
    
    Returns dict with:
        - resp_type: str
        - latency_ms: float
        - p_value: float (if statistical testing enabled)
        - test_results: dict (detailed test results)
    """
    

    inhib_thresh = max(MIN_INHIB_THRESHOLD, mu - K_SD_INHIB * sd)
    excit_thresh = mu + K_SD_EXCIT * sd

    min_bins_inhib = max(1, int(np.round(MIN_DURATION_INHIB_MS / BIN_SIZE_MS)))
    min_bins_excit = max(1, int(np.round(MIN_DURATION_EXCIT_MS / BIN_SIZE_MS)))
    
    search_mask = (t_centers >= MIN_LATENCY_MS) & (t_centers <= POST_TIME_MS)
    idxs = np.where(search_mask)[0]
    
    below = rate_smooth < inhib_thresh
    above = rate_smooth > excit_thresh
    
    # Detect threshold crossings
    inh_lat = None
    exc_lat = None
    
    # Inhibitory — requires (a) mu above MIN_MU, (b) sustained sub-threshold window,
    # AND (c) rate drops to at most INHIB_MAX_FRACTION * mu (relative criterion that
    # prevents noise-floor detections on low-firing cells).
    if mu >= MIN_MU:
        for i in idxs:
            if i + min_bins_inhib > len(below):
                break
            if below[i] and np.all(below[i:i+min_bins_inhib]):
                window_min = np.min(rate_smooth[i:i + min_bins_inhib])
                if window_min <= INHIB_MAX_FRACTION * mu:   # meaningful fractional drop
                    inh_lat = t_centers[i]
                    break

    # Excitatory — sustained point-threshold crossing
    for i in idxs:
        if i + min_bins_excit > len(above):
            break
        if above[i] and np.all(above[i:i+min_bins_excit]):
            exc_lat = t_centers[i]
            break

    # Excitatory fallback — mean-window criterion for broad responses where
    # baseline σ is large and the point threshold is hard to reach.
    if exc_lat is None:
        win_mask = (t_centers >= MEAN_WINDOW_MS[0]) & (t_centers <= MEAN_WINDOW_MS[1])
        if win_mask.any() and rate_smooth[win_mask].mean() > mu + K_MEAN_EXCIT * sd:
            for i in idxs:
                if rate_smooth[i] > mu:
                    exc_lat = t_centers[i]
                    break

    # Determine response type
    if inh_lat is None and exc_lat is None:
        return {
            'resp_type': 'none',
            'latency_ms': np.nan,
            'p_value': 1.0,
            'test_results': {},
            'significant': False
        }
    elif exc_lat is None:
        resp_type = 'inhibitory'
        latency = inh_lat
    elif inh_lat is None:
        resp_type = 'excitatory'
        latency = exc_lat
    else:
        # Both thresholds crossed — pick the one with the larger effect size
        # (peak z-score above baseline for excitation, trough for inhibition)
        # to avoid "earliest latency wins" mis-classifications.
        window_ms = 50.0   # look 50 ms ahead of each detected latency
        w_bins    = max(1, int(np.round(window_ms / BIN_SIZE_MS)))

        exc_i = np.argmin(np.abs(t_centers - exc_lat))
        inh_i = np.argmin(np.abs(t_centers - inh_lat))

        exc_effect = (np.max(rate_smooth[exc_i: exc_i + w_bins]) - mu) / sd if sd > 0 else 0.0
        inh_effect = (mu - np.min(rate_smooth[inh_i: inh_i + w_bins])) / sd if sd > 0 else 0.0

        if exc_effect >= inh_effect:
            resp_type = 'excitatory'
            latency   = exc_lat
        else:
            resp_type = 'inhibitory'
            latency   = inh_lat

    if latency > MAX_RESP_LATENCY_MS:
            return {
                'resp_type': 'none',
                'latency_ms': np.nan,
                'p_value': 1.0,
                'test_results': {},
                'significant': False
            }

    # Statistical testing
    if ENABLE_STATISTICAL_TESTS:
        # Get response window spike counts
        lat_idx = np.argmin(np.abs(t_centers - latency))
        if resp_type == 'excitatory':
            response_end_idx = min(lat_idx + min_bins_excit, len(spike_counts_all))
        else:
            response_end_idx = min(lat_idx + min_bins_inhib, len(spike_counts_all))
        
        spike_counts_response = spike_counts_all[lat_idx:response_end_idx]
        
        # Run statistical tests
        stats_result = test_response_significance(
            rate_smooth, spike_counts_baseline, spike_counts_response,
            mu, np.std(spike_counts_baseline), resp_type
        )
        
        return {
            'resp_type': resp_type,
            'latency_ms': latency,
            'p_value': stats_result['combined_p'],
            'test_results': stats_result['test_results'],
            'significant': stats_result['significant'],
            'alpha_corrected': stats_result['alpha_corrected']
        }
    else:
        return {
            'resp_type': resp_type,
            'latency_ms': latency,
            'p_value': 0.0,
            'test_results': {},
            'significant': True
        }


def detect_latency_mad(rate_smooth):
    """
    MAD-based variant of detect_latency for comparing scale estimates.

    Uses the baseline **median** as the location estimate and the median
    absolute deviation (scaled by 1.4826) as the dispersion estimate, so
    both are robust to outlier spikes.  Statistical tests are not re-run;
    only threshold-based detection is performed.

    Parameters
    ----------
    rate_smooth : array
        Smoothed firing rate (same array passed to detect_latency)

    Returns
    -------
    dict
        resp_type, latency_ms, mu_mad (baseline median), sd_mad
    """
    baseline_mask = (t_centers >= -BASELINE_PRE_MS) & (t_centers < 0)
    baseline_vals = rate_smooth[baseline_mask]
    mu_mad = float(np.median(baseline_vals))
    sd_mad = 1.4826 * np.median(np.abs(baseline_vals - mu_mad))

    inhib_thresh = max(MIN_INHIB_THRESHOLD, mu_mad - K_SD_INHIB * sd_mad)
    excit_thresh = mu_mad + K_SD_EXCIT * sd_mad

    min_bins_inhib = max(1, int(np.round(MIN_DURATION_INHIB_MS / BIN_SIZE_MS)))
    min_bins_excit = max(1, int(np.round(MIN_DURATION_EXCIT_MS / BIN_SIZE_MS)))

    search_mask = (t_centers >= MIN_LATENCY_MS) & (t_centers <= POST_TIME_MS)
    idxs = np.where(search_mask)[0]

    below = rate_smooth < inhib_thresh
    above = rate_smooth > excit_thresh

    inh_lat = None
    exc_lat = None

    if mu_mad >= MIN_MU:
        for i in idxs:
            if i + min_bins_inhib > len(below):
                break
            if below[i] and np.all(below[i:i + min_bins_inhib]):
                window_min = np.min(rate_smooth[i:i + min_bins_inhib])
                if window_min <= INHIB_MAX_FRACTION * mu_mad:
                    inh_lat = t_centers[i]
                    break

    for i in idxs:
        if i + min_bins_excit > len(above):
            break
        if above[i] and np.all(above[i:i + min_bins_excit]):
            exc_lat = t_centers[i]
            break

    if exc_lat is None:
        win_mask = (t_centers >= MEAN_WINDOW_MS[0]) & (t_centers <= MEAN_WINDOW_MS[1])
        if win_mask.any() and rate_smooth[win_mask].mean() > mu_mad + K_MEAN_EXCIT * sd_mad:
            for i in idxs:
                if rate_smooth[i] > mu_mad:
                    exc_lat = t_centers[i]
                    break

    _no_resp = {'resp_type': 'none', 'latency_ms': np.nan, 'mu_mad': mu_mad, 'sd_mad': sd_mad}
    if inh_lat is None and exc_lat is None:
        return _no_resp
    elif exc_lat is None:
        resp_type, latency = 'inhibitory', inh_lat
    elif inh_lat is None:
        resp_type, latency = 'excitatory', exc_lat
    else:
        window_ms = 50.0
        w_bins = max(1, int(np.round(window_ms / BIN_SIZE_MS)))
        exc_i = np.argmin(np.abs(t_centers - exc_lat))
        inh_i = np.argmin(np.abs(t_centers - inh_lat))
        exc_effect = (np.max(rate_smooth[exc_i:exc_i + w_bins]) - mu_mad) / sd_mad if sd_mad > 0 else 0.0
        inh_effect = (mu_mad - np.min(rate_smooth[inh_i:inh_i + w_bins])) / sd_mad if sd_mad > 0 else 0.0
        if exc_effect >= inh_effect:
            resp_type, latency = 'excitatory', exc_lat
        else:
            resp_type, latency = 'inhibitory', inh_lat

    if latency > MAX_RESP_LATENCY_MS:
        return _no_resp

    return {'resp_type': resp_type, 'latency_ms': latency, 'mu_mad': mu_mad, 'sd_mad': sd_mad}


### For testing
### Compute response latency based on 3 different methods:
###     1. SD-based (K_SD_inhib={K_SD_INHIB}, K_SD_excit={K_SD_EXCIT})"
###     2. Percentage-based ({INHIB_PERCENT_DROP}% drop)")
###     3. Percentile-based ({INHIB_PERCENTILE}th percentile)")

# def calculate_thresholds_all_methods(rate_smooth, mu, sd):
#     """Calculate inhibitory thresholds for methods 1, 2, and 3."""
#     baseline_mask = (t_centers >= -PRE_TIME_MS) & (t_centers < 0)
#     baseline_values = rate_smooth[baseline_mask]
    
#     thresholds = {}
    
#     # Method 1: SD-based
#     thresholds[1] = {
#         'name': 'SD-based',
#         'inhib': max(MIN_INHIB_THRESHOLD, mu - K_SD_INHIB * sd),
#         'excit': mu + K_SD_EXCIT * sd
#     }
    
#     # Method 2: Percentage-based
#     thresholds[2] = {
#         'name': 'Percentage-based',
#         'inhib': max(MIN_INHIB_THRESHOLD, mu * (1 - INHIB_PERCENT_DROP/100)),
#         'excit': mu + K_SD_EXCIT * sd
#     }
    
#     # Method 3: Percentile-based
#     thresholds[3] = {
#         'name': 'Percentile-based',
#         'inhib': max(MIN_INHIB_THRESHOLD, np.percentile(baseline_values, INHIB_PERCENTILE)),
#         'excit': mu + K_SD_EXCIT * sd
#     }
    
#     return thresholds


# def detect_latency_all_methods(rate_smooth, mu, sd, spike_counts_all, t_centers):
#     """
#     Detect latency using all 3 methods with statistical testing.
#     """
#     baseline_mask = (t_centers >= -PRE_TIME_MS) & (t_centers < 0)
#     spike_counts_baseline = spike_counts_all[baseline_mask]
    
#     thresholds = calculate_thresholds_all_methods(rate_smooth, mu, sd)
#     results = {}
    
#     for method_id in [1, 2, 3]:
#         thresh = thresholds[method_id]
#         result = detect_latency(
#             rate_smooth, mu, sd
#             spike_counts_all, spike_counts_baseline
#         )
        
#         results[method_id] = {
#             'resp_type': result['resp_type'],
#             'latency_ms': result['latency_ms'],
#             'inhib_thresh': thresh['inhib'],
#             'excit_thresh': thresh['excit'],
#             'method_name': thresh['name'],
#             'p_value': result['p_value'],
#             'test_results': result['test_results'],
#             'significant': result['significant']
#         }
    
#     return results


# ============================================================
# ADAPTIVE SMOOTHING: Higher smoothing for lower baseline rates
# ============================================================
def plot_psth(t_centers, rate, rate_smooth, mu, sd, results, cluster_id, stim, dc, tid,
              figure_dir, fmt='png', dpi=150):
    """
    Plot and save a single PSTH figure.

    Parameters
    ----------
    t_centers : array   Time axis in ms
    rate       : array  Raw (unsmoothed) firing rate in Hz
    rate_smooth: array  Smoothed firing rate in Hz
    mu, sd     : float  Baseline mean and std of smoothed rate
    results    : dict   Output of detect_latency (resp_type, latency_ms, p_value, significant)
    cluster_id, stim, dc, tid : identifiers for title and filename
    figure_dir : str or Path  Root figures directory
    fmt, dpi   : output format and resolution
    """
    resp_type  = results['resp_type']
    latency_ms = results['latency_ms']
    p_value    = results.get('p_value', None)

    excit_thresh = mu + K_SD_EXCIT * sd
    inhib_thresh = max(MIN_INHIB_THRESHOLD, mu - K_SD_INHIB * sd)

    RESP_COLOR = {'excitatory': '#E63946', 'inhibitory': '#457B9D', 'none': 'gray'}
    color = RESP_COLOR.get(resp_type, 'gray')

    fig, ax = plt.subplots(figsize=(9, 4))

    # Raw rate as light bars
    ax.bar(t_centers, rate, width=(t_centers[1] - t_centers[0]),
           color='lightgray', alpha=0.6, label='Raw rate')

    # Smoothed rate
    ax.plot(t_centers, rate_smooth, color='k', linewidth=1.8, label='Smoothed rate', zorder=3)

    # Baseline window shading
    ax.axvspan(-PRE_TIME_MS, 0, color='lightyellow', alpha=0.5, zorder=0)

    # Threshold lines
    ax.axhline(mu, color='gray', linestyle='--', linewidth=1.2, alpha=0.8, label=f'Baseline ({mu:.1f} Hz)')
    ax.axhline(excit_thresh, color='#E63946', linestyle=':', linewidth=1.2,
               alpha=0.7, label=f'+{K_SD_EXCIT}σ ({excit_thresh:.1f} Hz)')
    ax.axhline(inhib_thresh, color='#457B9D', linestyle=':', linewidth=1.2,
               alpha=0.7, label=f'−{K_SD_INHIB}σ ({inhib_thresh:.1f} Hz)')

    # Stimulus onset
    ax.axvline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.6, label='Stim onset')

    # Latency marker
    if not np.isnan(latency_ms):
        ax.axvline(latency_ms, color=color, linestyle='-', linewidth=1.8,
                   alpha=0.9, label=f'Latency {latency_ms:.1f} ms')

    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=11)
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)

    sig_str = ''
    if p_value is not None:
        sig_str = f'  p={p_value:.4f}'
        if p_value < 0.001:
            sig_str += ' (***)'
        elif p_value < 0.01:
            sig_str += ' (**)'
        elif p_value < 0.05:
            sig_str += ' (*)'

    ax.set_title(
        f'Cluster {cluster_id}  |  {stim}  DC={dc}  Train {tid}\n'
        f'Response: {resp_type}{sig_str}',
        fontsize=11, fontweight='bold', color=color
    )
    ax.legend(fontsize=8, loc='upper right', framealpha=0.7)

    out_dir = Path(figure_dir) / 'psth' / str(cluster_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f'{cluster_id}_{stim.replace("+", "_")}_{dc}Hz_train{tid}.{fmt}'
    fig.tight_layout()
    fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def get_adaptive_smooth_sd(baseline_rate):
    """
    Calculate appropriate smoothing based on baseline firing rate.
    Low firing rates need more smoothing to reduce noise.
    
    Parameters
    ----------
    baseline_rate : float
        Baseline firing rate in Hz
    
    Returns
    -------
    smooth_sd : float
        Smoothing sigma in bins
    """
    if baseline_rate < 3.0:
        return 5.0  # ~2.5 ms smoothing
    elif baseline_rate < 10.0:
        return 3.0  # ~1.5 ms smoothing
    else:
        return 10 #2.0  # ~1.0 ms smoothing


# ============================================================
# FOR Z-SCORE COMPUTATION
# ============================================================
def sort_key(item):
    data = item[1]
    resp_type = data.get('response_type', 'unknown')
    latency = data.get('latency', 999)
    # excitatory=0 (top), inhibitory=1 (bottom)
    type_order = 0 if resp_type == 'excitatory' else 1
    # Negate latency so larger values (slower) come first
    return (type_order, -latency)


# ============================================================
# LOADING BIN OBJ WITH PICKLE PROTOCOL INTO DATAFRAME
# Function from the Omarre analysis pipeline
# ============================================================
def load_obj_as_df(name):
    """
        Generic function to load a bin obj with pickle protocol

    Input :
        - name (str) : path to where the obj is
    Output :
        - (python object) : loaded object
        
    Possible mistakes :
        - Wrong path 
    """

    name = str(name)
    if os.path.dirname(os.path.normpath(name)) != '':
        os.makedirs(os.path.dirname(os.path.normpath(name)), exist_ok=True)
    else:
        name = os.path.join(os.getcwd(),os.path.normpath(name))
    if str(name)[-4:]!='.pkl':
        name += '.pkl'
    with open(os.path.normpath(name), 'rb') as f:
        tmp = pickle.load(f)
        return pd.DataFrame.from_dict(tmp, orient = "index")
    

# ============================================================
# EXTRACT ON/OFF CELL TYPE FROM THE baden_type DICTIONNARY
# ============================================================
def extract_cell_type(cell_type_value):
    """
    Extract ON / OFF / ON-OFF from a cell_type dict or string.

    Handles:
      - dict with a 'name' key  → e.g. {'id': 3, 'name': 'OFF suppression 2'}
      - plain string             → e.g. 'ON sustained'
      - None / NaN               → returns None
    """

    if cell_type_value is None:
        return None
    # Unwrap dict
    if isinstance(cell_type_value, dict):
        name = cell_type_value.get('name', '')
    elif isinstance(cell_type_value, str):
        name = cell_type_value
    else:
        return None   # unexpected type (float NaN, etc.)

    # Check ON-OFF first so it isn't caught by the bare ON / OFF checks
    if 'ON-OFF' in name or 'ON/OFF' in name:
        return 'ON-OFF'
    elif 'OFF' in name:
        return 'OFF'
    elif "ON" in name:
        return 'ON'
    return None


# ============================================================
# PRECOMPUTE WIDE (PIVOTED) DATAFRAMES  ← insert once, reuse everywhere
# ============================================================
# The resulting `df_wide` has one row per (cluster_id, dc) and columns like 
# latency_ms_pa / latency_ms_pl, response_type_pa / response_type_pl, etc.

def build_wide(source_df, exclude_dc=37):
    """
    Pivot a long-format results DataFrame (columns: cluster_id, dc, stim, ...)
    into a wide format indexed by (cluster_id, dc) with _pa/_pl suffixes.

    Parameters
    ----------
    source_df : pd.DataFrame
        Long-format DF with at least columns [cluster_id, dc, stim, ...].
    exclude_dc : int or None
        DC value to drop (default 37).  Pass None to keep all.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame indexed by (cluster_id, dc).
    """
    data = source_df.copy()
    if exclude_dc is not None:
        data = data[data["laser_prr"] != exclude_dc]

    # Separate the two conditions
    pa = (data[data["stim"] == "PA"]
          .drop(columns="stim")
          .set_index(["cluster_id", "laser_prr"])
          .add_suffix("_pa"))

    pl = (data[data["stim"] == "PA+light"]
          .drop(columns="stim")
          .set_index(["cluster_id", "laser_prr"])
          .add_suffix("_pl"))

    wide = pd.concat([pa, pl], axis=1, join="outer")
    return wide

def build_wide_drug(source_df, exclude_dc=37):
    ### For session 250904_A
    
    """
    Pivot a long-format results DataFrame (columns: cluster_id, dc, stim, ...)
    into a wide format indexed by (cluster_id, dc) with _pa/_pl suffixes.

    Parameters
    ----------
    source_df : pd.DataFrame
        Long-format DF with at least columns [cluster_id, dc, stim, ...].
    exclude_dc : int or None
        DC value to drop (default 37).  Pass None to keep all.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame indexed by (cluster_id, dc).
    """
    data = source_df.copy()
    if exclude_dc is not None:
        data = data[data["dc"] != exclude_dc]

    # Separate the two conditions
    pa = (data[data["stim"] == "PA"]
          .drop(columns="stim")
          .set_index(["cluster_id", "dc"])
          .add_suffix("_pa"))

    drug = (data[data["stim"] == "PA_LAP4"]
          .drop(columns="stim")
          .set_index(["cluster_id", "dc"])
          .add_suffix("_lap4"))

    wide = pd.concat([pa, pl], axis=1, join="outer")
    return wide


### ── Helper functions for the cell typing part ──────────────────────────────
CELL_TYPES  = ["ON", "OFF", "ON-OFF", "Unknown"]
RESP_LABELS = ["Excitatory", "Inhibitory", "No response"]
RESP_COLORS = {
    "Excitatory":  "#E63946",
    "Inhibitory":  "#457B9D",
    "No response": "#FFDA37",
}

def majority_response(series):
    """Return the most frequent non-null value, or None."""
    vals = series.dropna()
    vals = [v for v in vals if isinstance(v, str)]
    if not vals:
        return None
    return pd.Series(vals).mode()[0]

def norm_resp(v):
    if not isinstance(v, str):
        return "No response"
    return v.capitalize()   # 'excitatory' → 'Excitatory', etc.

def count_table(df_cells, resp_col):
    """Return DataFrame[cell_type × response_type] with counts."""
    ct = (
        df_cells.groupby(["cell_type", resp_col])
        .size()
        .unstack(fill_value=0)
        .reindex(index=CELL_TYPES, columns=RESP_LABELS, fill_value=0)
    )
    return ct

def build_counts(df_source, per_cell=True):
    """
    Build a count + percentage table from df_merged_wide (or a DC slice).
    If per_cell=True, collapse to one row per cluster via majority vote first.
    Returns (counts_pa, counts_pl, pct_pa, pct_pl).
    """
    if per_cell:
        df = (
            df_source.reset_index()
            .groupby("cluster_id")
            .agg(
                response_type_pa=("response_type_pa", majority_response),
                response_type_pl=("response_type_pl", majority_response),
                baden_type=("baden_type", "first"),
            )
            .reset_index()
        )
    else:
        df = df_source.reset_index().copy()

    df["cell_type"] = df["baden_type"].apply(lambda bt: extract_cell_type(bt) or "Unknown")
    df["resp_pa"]   = df["response_type_pa"].apply(norm_resp)
    df["resp_pl"]   = df["response_type_pl"].apply(norm_resp)

    def pivot(resp_col):
        ct = (
            df.groupby(["cell_type", resp_col])
            .size()
            .unstack(fill_value=0)
            .reindex(index=CELL_TYPES, columns=RESP_LABELS, fill_value=0)
        )
        # Drop all-zero rows (cell types not present)
        ct = ct.loc[ct.sum(axis=1) > 0]
        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        return ct, pct

    counts_pa, pct_pa = pivot("resp_pa")
    counts_pl, pct_pl = pivot("resp_pl")
    return counts_pa, counts_pl, pct_pa, pct_pl

def print_summary(label, counts_pa, counts_pl, pct_pa, pct_pl):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for stim, counts, pct in [("PA", counts_pa, pct_pa), ("PA+light", counts_pl, pct_pl)]:
        print(f"\n  [{stim}]")
        header = f"  {'Cell type':<12}" + "".join(f"{r:>16}" for r in RESP_LABELS)
        print(header)
        print("  " + "-" * (12 + 16 * len(RESP_LABELS)))
        for ct in counts.index:
            total = counts.loc[ct].sum()
            row = f"  {ct:<12}"
            for r in RESP_LABELS:
                n = int(counts.loc[ct, r])
                p = pct.loc[ct, r]
                row += f"  {n:>4} ({p:>5.1f}%)  "
            print(row + f"  [n={total}]")


def plot_grouped_bars(axes_row, counts_pa, counts_pl, pct_pa, pct_pl, title_suffix=""):
    """Fill a pair of axes with grouped bar charts (counts + % annotations)."""
    for ax, (counts, pct, stim_label) in zip(
        axes_row,
        [
            (counts_pa, pct_pa, "PA"),
            (counts_pl, pct_pl, "PA + light"),
        ],
    ):
        present_types = list(counts.index)
        n_ct   = len(present_types)
        n_resp = len(RESP_LABELS)
        x      = np.arange(n_ct)
        width  = 0.22
        offsets = np.linspace(-(n_resp - 1) / 2, (n_resp - 1) / 2, n_resp) * width

        for resp, offset in zip(RESP_LABELS, offsets):
            heights = [int(counts.loc[ct, resp]) for ct in present_types]
            pcts    = [pct.loc[ct, resp] for ct in present_types]
            bars = ax.bar(
                x + offset, heights, width,
                label=resp,
                color=RESP_COLORS[resp],
                edgecolor="k",
                linewidth=0.6,
                alpha=0.88,
            )
            for bar, h, p in zip(bars, heights, pcts):
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.05,
                        f"{h}\n({p:.0f}%)",
                        ha="center", va="bottom",
                        fontsize=6.5, fontweight="bold", color="#222",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(present_types, fontsize=10)
        ax.set_xlabel("Cell type", fontsize=10, fontweight="bold")
        ax.set_ylabel("Number of cells", fontsize=10, fontweight="bold")
        ax.set_title(f"{stim_label}  {title_suffix}", fontsize=11, fontweight="bold")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)