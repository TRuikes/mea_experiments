import numpy as np

# ============================================================
# STATISTICAL TESTING PARAMETERS
# ============================================================

# Enable/disable statistical testing
ENABLE_STATISTICAL_TESTS = True  # Set to False to use only threshold-based detection

# Choose which statistical tests to run (can select multiple)
STATISTICAL_TESTS = {
    'bootstrap': True,       # Bootstrap resampling test
    't_test': False,         # Welch's t-test
    'permutation': False,    # Permutation test
    'sliding_window': False  # Sliding window t-test
}

# Statistical test parameters
SIGNIFICANCE_ALPHA = 0.05  # p-value threshold for significance
N_BOOTSTRAP = 1000         # Number of bootstrap iterations
N_PERMUTATIONS = 1000      # Number of permutation iterations
BONFERRONI_CORRECTION = True  # Apply Bonferroni correction for multiple comparisons

# Sliding window parameters
WINDOW_SIZE_MS = 50.0      # Size of sliding window in ms
WINDOW_STEP_MS = 5.0       # Step size for sliding window in ms

# ============================================================
# RESPONSE DETECTION PARAMETERS
# ============================================================
BIN_SIZE_MS     = 0.5
PRE_TIME_MS     = 200.0  # Pre-stimulus window displayed / collected (ms)
BASELINE_PRE_MS = 100.0  # Portion of pre-stimulus used for µ/σ computation (ms before 0)
POST_TIME_MS    = 500.0  # Response: from 0 ms to 500 ms

# Different duration requirements for excitation vs inhibition
MIN_DURATION_EXCIT_MS = 10.0
MIN_DURATION_INHIB_MS = 30.0

# Minimum latency - ignore responses before this time
MIN_LATENCY_MS = 10.0

# Parameters for the 3 methods
K_SD_EXCIT = 2.0
K_SD_INHIB = 2.0
INHIB_PERCENT_DROP = 30
INHIB_PERCENTILE = 5

MIN_INHIB_THRESHOLD = 0.5
MIN_BASELINE_HZ = 1.0
MIN_MU = 5.0  # Minimum baseline for detecting inhibition

MAX_RESP_LATENCY_MS = 250 # Disregard responses that have a higher latency than this (ms)

# Mean-window excitatory fallback — catches broad responses where baseline σ is
# large enough that the point threshold (µ + K_SD_EXCIT*sd) is rarely reached.
K_MEAN_EXCIT   = 1.5           # mean post-stim rate must exceed µ + K_MEAN_EXCIT*sd
MEAN_WINDOW_MS = (10.0, 200.0) # (start_ms, end_ms) for the mean-based criterion

# Inhibitory response must bring the rate to at most this fraction of baseline mu.
# E.g. 0.6 means the smoothed rate must drop below 60 % of the baseline mean
# for at least MIN_DURATION_INHIB_MS — guards against noise-floor detections.
INHIB_MAX_FRACTION = 0.6

# ============================================================
# TIME AXIS
# ============================================================

t_edges = np.arange(-PRE_TIME_MS, POST_TIME_MS + BIN_SIZE_MS, BIN_SIZE_MS)
t_centers = t_edges[:-1] + BIN_SIZE_MS / 2

# ============================================================
# PSTH PLOTTING
# ============================================================
PLOT_PSTH = True         # Set to True to save one figure per (cluster × train)
PSTH_PLOT_FORMAT = 'png'  # 'png' or 'pdf'
PSTH_PLOT_DPI = 150