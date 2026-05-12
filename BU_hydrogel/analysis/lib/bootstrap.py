import numpy as np


class BootstrapOutput:
    bins=None  # type: np.ndarray
    bin_size=None  # type: int
    binned_sp=None  # type: np.ndarray
    spike_times=None

    firing_rate=None  # type: np.ndarray
    firing_rate_ci_low=None  # type: np.ndarray
    firing_rate_ci_high=None  # type: np.ndarray

    baseline_firing_rate_mean=None
    baseline_firing_rate_max = None

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

    mean_response_if_not_sig=None
    max_response_if_not_sig = None


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

    def __init__(self):
        pass




