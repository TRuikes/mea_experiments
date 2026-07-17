import numpy as np
from scipy.stats import bootstrap
from typing import List
from sonogenetics.analysis.lib.poisson_rate_estimation import first_consecutive_run

class BootstrapOutput:
    def __init__(self,
                 bins: np.ndarray,
                 binned_sp: np.ndarray,
                 firing_rate: np.ndarray,
                 baseline_firing_rate_mean: float,
                 is_excited: bool,
                 is_inhibited: bool,
                 excitation_bins: np.ndarray | None,
                 excitation_max_fr: float | None,
                 excitation_start: float | None,
                 excitation_duration: float | None,
                 excitation_end: float | None,
                 inhibition_bins: np.ndarray | None,
                 inhibition_min_fr: float | None,
                 inhibition_start: float | None,
                 inhibition_duration: float | None,
                 inhibition_end: float | None,
                 firing_rate_ci_high: np.ndarray,
                 firing_rate_ci_low: np.ndarray,
                 has_bootstrap: bool,
    ):
        self.bins = bins
        self.binned_sp = binned_sp
        self.firing_rate = firing_rate
        self.baseline_firing_rate_mean = baseline_firing_rate_mean
        self.is_excited = is_excited
        self.excitation_bins = excitation_bins
        self.excitation_max_fr = excitation_max_fr
        self.excitation_start = excitation_start
        self.excitation_duration = excitation_duration
        self.excitation_end = excitation_end
        self.is_inhibited = is_inhibited
        self.inhibition_bins = inhibition_bins
        self.inhibition_min_fr = inhibition_min_fr
        self.inhibition_start = inhibition_start
        self.inhibition_duration = inhibition_duration
        self.inhibition_end = inhibition_end
        self.firing_rate_ci_high = firing_rate_ci_high
        self.firing_rate_ci_low = firing_rate_ci_low
        self.has_bootstrap = has_bootstrap


    single_value_stats = [
        'baseline_firing_rate_mean',
        'baseline_firing_rate_max',
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
        'mean_response_if_not_sig',
        'max_response_if_not_sig',
    ]

    has_data=False
    reason='none'

    def get(self, name):
        assert hasattr(self, name), f'{name} not in attributes'
        return getattr(self, name)


def detect_significant_modulation_bootstrap(
        bin_centres, binned_sp: np.ndarray, baseline_idx, min_duration_ms, stepsize_ms, binwidth_ms
):
    # Bootstrap confidence intervals for each bin
    response_window = [0, 200]
    n_spikes_per_bin: np.ndarray = np.sum(binned_sp, axis=0)
    idx: np.ndarray = np.where(n_spikes_per_bin > 0)[0]
    n_bins = bin_centres.size

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

    ci_baseline: List[float] = [float(np.nanmean(ci_low[baseline_idx])), float(np.nanmean(ci_high[baseline_idx]))]
    firing_rate = np.mean(binned_sp, axis=0) / (binwidth_ms / 1000)
    firing_rate_ci_high = ci_high / (binwidth_ms / 1000)
    firing_rate_ci_low = ci_low / (binwidth_ms / 1000)

    # # Detect which bins are decrease relative to baseline
    in_idx = np.where((ci_high < ci_baseline[0]) & (bin_centres >= response_window[0]) &
                      (bin_centres < response_window[1]))[0]
    in_idx = first_consecutive_run(in_idx, int(min_duration_ms / min_duration_ms))

    is_inhibited = True if in_idx is not None else False
    inhibition_bins = in_idx if is_inhibited else None
    inhibition_min_fr = np.min(firing_rate[in_idx]) if is_inhibited else None
    inhibition_start = bin_centres[in_idx[0]] if is_inhibited else None
    inhibition_end = bin_centres[in_idx[-1]] if is_inhibited else None
    inhibition_duration = inhibition_end - inhibition_start if is_inhibited else None

    # Extract excitation stats
    # # Detect which bins are increased relative to baseline
    ex_idx = np.where((ci_low > ci_baseline[1]) & (bin_centres >= response_window[0]) &
                      (bin_centres < response_window[1]))[0]
    ex_idx = first_consecutive_run(ex_idx, int(min_duration_ms / stepsize_ms),
                                   occupancy_threshold=0.8)

    is_excited = True if ex_idx is not None else False
    excitation_bins = ex_idx if is_excited else None
    excitation_max_fr = np.max(firing_rate[is_excited]) if is_excited else None
    excitation_start = bin_centres[ex_idx[0]] if is_excited else None
    excitation_end = bin_centres[ex_idx[-1]] if is_excited else None
    excitation_duration = excitation_end - excitation_start if is_excited else None

    return BootstrapOutput(
        bins=bin_centres,
        binned_sp=binned_sp,
        firing_rate=firing_rate,
        baseline_firing_rate_mean=np.mean(firing_rate[baseline_idx]),
        is_excited=is_excited,
        is_inhibited=is_inhibited,
        excitation_bins=excitation_bins,
        excitation_max_fr=excitation_max_fr,
        excitation_start=excitation_start,
        excitation_duration=excitation_duration,
        excitation_end=excitation_end,
        inhibition_bins=inhibition_bins,
        inhibition_min_fr=inhibition_min_fr,
        inhibition_start=inhibition_start,
        inhibition_duration=inhibition_duration,
        inhibition_end=inhibition_end,
        firing_rate_ci_low=firing_rate_ci_low,
        firing_rate_ci_high=firing_rate_ci_high,
        has_bootstrap=has_btrp
    )



