import pandas as pd
from pv_chip_aarhus.preprocessing.lib.filepaths import FilePaths
import numpy as np

def get_stim_per_channel(stim_file):
    channel_tables = {}
    current_channel = None
    with open(stim_file, "r") as file:
        for line in file:
            clean_line = line.strip()

            # 1. Skip empty lines or global file headers
            if not clean_line or clean_line.startswith(
                    ("Multi Channel", "ASCII", "channels:", "output mode:", "format:")
            ):
                continue

            # 2. Detect a new channel block
            if clean_line.startswith("channel:"):
                # Save the previous channel's data before moving to the new one
                if current_channel and current_rows:
                    channel_tables[current_channel] = pd.DataFrame(
                        current_rows, columns=headers
                    )

                # Reset variables for the new channel
                current_channel = clean_line.split(":")[1].strip()
                current_rows = []
                headers = None
                continue

            # 3. Capture the column headers for the current table
            if clean_line.startswith("value"):
                headers = clean_line.split("\t")
                continue

            # 4. Collect data rows (split by tab characters)
            if current_channel:
                row_data = clean_line.split("\t")
                # Convert string numbers to actual integers/floats
                row_data = [
                    float(val) if "." in val else int(val) for val in row_data
                ]
                current_rows.append(row_data)

        # Don't forget to save the very last channel block after the loop finishes
        if current_channel and current_rows:
            channel_tables[current_channel] = pd.DataFrame(
                current_rows, columns=headers
            )
    return channel_tables


def extract_trial_data(filepaths: FilePaths):
    df = pd.DataFrame()

    trial_i = 0

    for rec_nr, rec_info in filepaths.recording_table.iterrows():

        stim_per_channel = get_stim_per_channel(filepaths.stim_dir / rec_info['stim_file'])

        has_laser = rec_info['stimsource'] in ['L', 'B']
        has_pchrome = rec_info['stimsource'] in ['P', 'B']

        laser_channel = rec_info['laser_ch'].split('ch')[1]

        if has_laser:
            laser_stim_table = stim_per_channel[laser_channel]
            print(laser_stim_table.shape, rec_nr)
            for i, r in laser_stim_table.iterrows():
                if i == 0:
                    continue

                if r['repeat'] == 1:
                    iti = r['time'].iloc[0] / 1e3
                    df.at[trial_i, 'iti'] = iti
                    trial_i += 1


                if r['repeat'] > 1:
                    on_duration = r['time'].iloc[1] / 1e3  # ms
                    off_duration = r['time'].iloc[2] / 1e3
                    df.at[trial_i, 'laser_on_duration'] = on_duration
                    df.at[trial_i, 'laser_off_duration'] = off_duration
                    df.at[trial_i, 'laser_repeats'] = r['repeat']


        elif has_pchrome:
            stim_table = stim_per_channel['9']
        else:
            raise ValueError('no stim?')


        n_param_sets = []



        break

    df.to_csv(filepaths.proc_pp_trials)