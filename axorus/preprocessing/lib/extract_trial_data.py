import pandas as pd
from axorus.preprocessing.lib.filepaths import FilePaths


def extract_trial_data(filepaths: FilePaths):

    df = pd.read_csv(filepaths.raw_trials, index_col=0, header=0)

    if df.shape[1] == 0:  # use another delimiter
        df = pd.read_csv(
            filepaths.raw_trials, index_col=0, header=0,
            delimiter=';'
        )

    trial_i = 0
    for i, r in df.iterrows():
        trial_id = f''