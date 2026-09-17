from tqdm import tqdm
import numpy as np
from typing import List, Tuple, cast, Any, Dict, Union
from pathlib import Path

import utils
from sonogenetics.analysis.lib.analysis_params import dataset_dir
from sonogenetics.analysis.data_list import data_list
from sonogenetics.analysis.lib.data_io import DataIO
from sonogenetics.analysis.lib.poisson_rate_estimation import detect_significant_modulation_poisson, PoissonOutput
from sonogenetics.analysis.lib.bootstrap import detect_significant_modulation_bootstrap

checkerboard_params = {
    'nb_checks_x': 30,
    'nb_checks_y': 30,
    'stimulus_frequency': 30,
    'nb_frames_by_sequence': 1200,  # Number of frames in each checkerboard sequence
}

def main():
    data_io = DataIO(dataset_dir)
    data_io.load_session('2026-09-09 mouse c57 758 Mekano6 C', load_pickle=False)

    rec_id = [r for r in data_io.recording_ids if 'checkerboard' in r][0]

    return



if __name__ == '__main__':
    main()