from pathlib import Path
import pandas as pd


def get_probe_layout(probe_name):

    # Define a dictionary to store the variables
    variables = {}

    file = r'F:\Axorus\ex_vivo_series_3' + f'\\{probe_name}.prb'

    # Open the file in read mode
    with open(file, 'r') as f:
        # Read the entire file content
        file_content = f.read()

    # Execute the file content as Python code
    exec(file_content, variables)

    probe_layout = variables['channel_groups'][1]['geometry']

    df = pd.DataFrame()
    for chnr, (x, y) in probe_layout.items():
        df.at[chnr + 1, 'x'] = x
        df.at[chnr + 1, 'y'] = y

    return df