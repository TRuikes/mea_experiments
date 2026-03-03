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
BIN_SIZE_MS  = 0.5
PRE_TIME_MS  = 200.0     # Baseline: from -200 ms to 0 ms
POST_TIME_MS = 500.0     # Response: from 0 ms to 500 ms

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

# ============================================================
# TIME AXIS
# ============================================================

t_edges = np.arange(-PRE_TIME_MS, POST_TIME_MS + BIN_SIZE_MS, BIN_SIZE_MS)
t_centers = t_edges[:-1] + BIN_SIZE_MS / 2