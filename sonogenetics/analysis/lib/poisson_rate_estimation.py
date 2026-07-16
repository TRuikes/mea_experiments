import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class PoissonOutput:
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

    def get(self, name):
        assert hasattr(self, name), f'{name} not in attributes'
        return getattr(self, name)


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






def detect_significant_modulation_poisson(bin_centres, binned_sp, baseline_idx, min_duration_ms,
                                  stepsize_ms):
    """
    POISSON BIN-BY-BIN SIGNIFICANCE ANALYSIS
    ----------------------------------------
    LOGIC:
    We model neuronal spiking as a Poisson process. When baseline firing rates are
    very low, non-parametric tests (like bootstrapping) fail due to zero-inflation.
    By fitting a parametric Poisson distribution to the baseline, we can calculate
    the exact probability of observing a low spike count by chance, even if that
    observed count is zero.

    GRAPHICAL CONCEPT:
          Baseline Window                 Post-Baseline (Test Window)
     [ | ||  |  |||  | || | ]           [    |         |        |    ] <-- Sparse Spikes
    ---------------------------       ---------------------------------
    Calculate mean spikes/bin          Test each bin individually:
    across all trials & bins:          Is the sum of spikes across trials (k)
          ==> lambda_base              unusually low relative to lambda_test?

          POISSON PROBABILITY MASS FUNCTION (PMF) FOR THE TEST BIN:
          P(X) ^
               |      * (Most likely spike count under Null)
               |     ***
               |    *****
               |   *******
               |  ********* *
               | *********** * * +-------------------> Spike Count (X)
                 [k]               [lambda_test]
                  |______________________|
                     p-value Area (CDF)
                     (Probability of getting <= k spikes by chance)

    MATH:
    1. Estimate the baseline rate (lambda_base) as the mean spikes per bin across
       all baseline bins and trials:
           lambda_base = Total Baseline Spikes / (Num Baseline Bins * Num Trials)

    2. Under the Null Hypothesis (no change in rate), the expected number of spikes
       in any single test bin, summed across all trials, is:
           Expected Spikes (lambda_test) = lambda_base * Num Trials

    3. For each post-baseline bin, we calculate the sum of observed spikes across
       all trials (k) and compute a one-tailed p-value using the Poisson Cumulative
       Distribution Function (CDF):
           p-value = P(X <= k | lambda_test) = sum_{i=0}^{k} [ (lambda_test^i * e^-lambda_test) / i! ]

    4. Bins with a p-value < alpha (e.g., 0.05) are flagged as significantly decreased.
    """

    # 1. Dynamically find all indices after the last baseline index
    last_baseline_idx = np.max(baseline_idx)
    test_idx = np.arange(last_baseline_idx + 1, binned_sp.shape[1])

    # 2. Calculate baseline statistics
    n_trials = binned_sp.shape[0]
    baseline_data = binned_sp[:, baseline_idx]
    mean_spikes_per_baseline_bin = np.mean(baseline_data)

    # Expected spikes in a single bin summed across all trials
    expected_spikes_per_bin = mean_spikes_per_baseline_bin * n_trials

    # 3. Initialize arrays to store results for each post-baseline bin
    n_test_bins = len(test_idx)
    p_values_dec = np.zeros(n_test_bins)
    p_values_inc = np.zeros(n_test_bins)

    significant_decrease = np.zeros(n_test_bins, dtype=bool)
    significant_increase = np.zeros(n_test_bins, dtype=bool)
    alpha = 0.05  # Note: If doing two-tailed, you could use alpha = 0.025 for each side

    for i, bin_col in enumerate(test_idx):
        # Sum the observed spikes in this specific bin across all trials
        observed_spikes = np.sum(binned_sp[:, bin_col])

        # --- Inhibition Test (one-tailed decrease) ---
        p_dec = stats.poisson.cdf(observed_spikes, expected_spikes_per_bin)
        p_values_dec[i] = p_dec
        significant_decrease[i] = p_dec < alpha

        # --- Excitation Test (one-tailed increase) ---
        # We subtract 1 because we want P(X >= observed_spikes)
        p_inc = stats.poisson.sf(observed_spikes - 1, expected_spikes_per_bin)
        p_values_inc[i] = p_inc
        significant_increase[i] = p_inc < alpha

    # 5. Calculate actual rates (Hz) for plotting
    dt = bin_centres[1] - bin_centres[0]
    firing_rate_hz = np.mean(binned_sp, axis=0) / dt
    baseline_rate_hz = mean_spikes_per_baseline_bin / dt

    # Extract inhibition stats
    bins_decrease = test_idx[significant_decrease]
    in_idx = first_consecutive_run(bins_decrease, int(min_duration_ms / stepsize_ms),
                                   occupancy_threshold=0.7) # indices into significant decrease

    is_inhibited = True if in_idx is not None else False
    inhibition_bins = bins_decrease if is_inhibited else None
    inhibition_min_fr = np.min(firing_rate_hz[bins_decrease]) if is_inhibited else None
    inhibition_start = bin_centres[in_idx[0]] if is_inhibited else None
    inhibition_end = bin_centres[bins_decrease[-1]] if is_inhibited else None
    inhibition_duration = inhibition_end - inhibition_start if is_inhibited else None

    # Extract excitation stats
    bins_increase = test_idx[significant_increase]
    ex_idx = first_consecutive_run(bins_increase, int(min_duration_ms / stepsize_ms),
                                   occupancy_threshold=0.7)
    is_excited = True if ex_idx is not None else False
    excitation_bins = bins_increase if is_excited else None
    excitation_max_fr = np.max(firing_rate_hz[bins_increase]) if is_excited else None
    excitation_start = bin_centres[ex_idx[0]] if is_excited else None
    excitation_end = bin_centres[bins_increase[-1]] if is_excited else None
    excitation_duration = excitation_end - excitation_start if is_excited else None


    # UNCOMMENT THIS DURING DEBUG MODE TO VISUALIZE THE DETECTION

    # 6. Plotting
    # plt.figure(figsize=(10, 5))
    #
    # # Plot the average firing rate (PSTH)
    # plt.plot(bin_centres, firing_rate_hz, color='black', label='Firing Rate (Hz)', lw=2)
    #
    # # Highlight baseline window
    # baseline_start_time = bin_centres[np.min(baseline_idx)]
    # baseline_end_time = bin_centres[last_baseline_idx]
    # plt.axvspan(baseline_start_time, baseline_end_time, color='gray', alpha=0.2, label='Baseline Window')
    #
    # if len(bins_decrease) > 0:
    #     plt.scatter(bin_centres[bins_decrease], firing_rate_hz[bins_decrease],
    #                 color='red', edgecolor='black', zorder=5, label='Significantly Decreased')
    #
    # # Add baseline reference line
    # plt.axhline(baseline_rate_hz, color='blue', linestyle='--', alpha=0.7, label='Baseline Mean')
    #
    # plt.title('Bin-by-Bin Poisson Significance Analysis')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Firing Rate (Hz)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return PoissonOutput(
        bins=bin_centres,
        binned_sp=binned_sp,
        firing_rate=firing_rate_hz,
        baseline_firing_rate_mean=baseline_rate_hz,
        is_excited=is_excited,
        is_inhibited=is_inhibited,
        excitation_bins=excitation_bins,
        excitation_max_fr=excitation_max_fr,
        excitation_start=excitation_start,
        excitation_end=excitation_end,
        excitation_duration=excitation_duration,
        inhibition_bins=inhibition_bins,
        inhibition_min_fr=inhibition_min_fr,
        inhibition_start=inhibition_start,
        inhibition_duration=inhibition_duration,
        inhibition_end=inhibition_end,
    )

