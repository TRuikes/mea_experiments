from kilosort import run_kilosort, DEFAULT_SETTINGS
from kilosort.io import load_probe
from pathlib import Path
import numpy as np


probe_file = r"E:\bu_hudrogel\2026-05-12 rat LE 1355 A\raw\256_100_30_mea_kilosort.json"
data_dir = r'E:\bu_hudrogel\2026-05-12 rat LE 1355 A dmd only\raw'
results_dir = r'E:\bu_hudrogel\2026-05-12 rat LE 1355 A dmd only\processed'

data_dir = Path(data_dir)
filenames = [f.as_posix() for f in data_dir.iterdir() if f.suffix == '.raw']


probe = load_probe(probe_file)

settings = DEFAULT_SETTINGS
settings['n_chan_bin'] = 256
settings['fs'] = 20000

ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
    run_kilosort(settings=settings, probe=probe, do_CAR=False, filename=filenames,
                 results_dir=results_dir, data_dtype='uint16')