import pandas as pd
from pv_chip_aarhus.preprocessing.lib.filepaths import FilePaths
import numpy as np

# Polychrome power stimulations are always conducted in this exact order.
POLYCHROME_POWERS = [12, 25, 50, 100]
LASER_POWERS = [10, 20, 30, 88]  # [mV] ,all area measured powers, 88 is max power

def get_stim_per_mcd_channel(stim_file):
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

        if rec_info['varied_param'] == 'flick':
            print(f'SKIPPING THE FLICK RECORDING')
            continue

        trial_offset = trial_i

        # the polychrome and laser are controlled by mcd, which also sends out trigger signals to mc rack
        # the laser trigger channel can vary between recordings, and is annotated in rec_info (Based on recording
        # names)
        # the polychrome channel is always ch9

        # Here we read out the MC stimulation file, so that we can align it with the recorded trigger file later
        # on.

        # Extract trigger signal per stimulation channel
        stim_per_mcd_channel = get_stim_per_mcd_channel(filepaths.stim_dir / rec_info['stim_file'])
        laser_channel = rec_info['laser_ch'].split('ch')[1]

        # Check if there is laser and/or polychrome stimulation
        has_laser = rec_info['stimsource'] in ['L', 'B']
        has_pchrome = rec_info['stimsource'] in ['P', 'B']


        if has_laser and has_pchrome:
            # Make sure each row of the table has the same time duration, so that we can simply align them
            laser_stim_table = stim_per_mcd_channel[laser_channel]
            pchr_stim_table = stim_per_mcd_channel['9']

            n_rows_laser = laser_stim_table.shape[0]
            n_rows_pchrome = pchr_stim_table.shape[0]
            assert n_rows_laser == n_rows_pchrome

            for row_i in range(n_rows_laser):
                time_laser = laser_stim_table.iloc[row_i]['time'].sum() * laser_stim_table.iloc[row_i]['repeat']
                time_pchrome = pchr_stim_table.iloc[row_i]['time'].sum() * pchr_stim_table.iloc[row_i]['repeat']
                assert time_laser == time_pchrome


        if has_laser:
            laser_stim_table = stim_per_mcd_channel[laser_channel]
            laser_tick = 0

            for i, r in laser_stim_table.iterrows():
                if i == 0:
                    assert r['repeat'] == 1
                    continue

                # In some stimfiles, the 1st and 2nd time columns are used, in others the 2nd and 3rd
                # here detect which columns are used
                idx = np.where(r['time'].values > 0)[0]

                if r['repeat'] == 1:
                    # For the inter trial intervals, there should be only a single time value
                    assert len(idx) == 1
                    iti = r['time'].iloc[idx[0]] / 1e3
                    df.at[trial_i, 'iti'] = iti
                    trial_i += 1


                if r['repeat'] > 1:
                    # The laser pulse is designed using 3 pairs of time and value. Those will describe the
                    # laser delay
                    # laser on
                    # laser off
                    # e.g.:
                    #   delay       on         off
                    #           ---------
                    #         |         |
                    # ---------         --------------------------------
                    #
                    # however not every stimulation row/type as all of those, in some cases the delay and off are missing

                    # In any case, there should be only 1 on value
                    idx = np.where(r['value'].values > 0)[0]
                    assert len(idx) == 1

                    # Now loop over the columns, to see if there is a delay before the first on
                    had_on = False
                    has_delay = False
                    laser_delay, laser_on_duration, laser_off_duration = None, None, None

                    for col_i in range(3):
                        t = r['time'].values[col_i]
                        v = r['value'].values[col_i]

                        if t == 0:  # if time = 0 there is no data in that column
                            continue

                        t = t / 1e3  # convert to [ms]

                        if v == 0:
                            if not had_on:  # if there is a v=0 column before a stimulation, there is an
                                    # onset delay
                                laser_delay = t
                                has_delay = True
                            elif had_on:  # if there is a v=0 column after a stimulation, there is an off
                                # off duration
                                laser_off_duration = t

                        elif v > 0:
                            assert had_on == False  # redundant, but a value for v can be occuring once in each row
                            laser_on_duration = t
                            had_on = True

                            if not has_delay:
                                laser_delay = 0

                    assert had_on

                    df.at[trial_i, 'stimsource'] = rec_info['stimsource']
                    df.at[trial_i, 'laser_delay'] = laser_delay
                    df.at[trial_i, 'laser_on_duration'] = laser_on_duration
                    df.at[trial_i, 'laser_off_duration'] = laser_off_duration
                    df.at[trial_i, 'laser_repeats'] = r['repeat']
                    df.at[trial_i, 'varied_param'] = rec_info['varied_param']

                    if rec_info['varied_param'] == 'pow':
                        df.at[trial_i, 'laser_power'] = LASER_POWERS[laser_tick]
                    else:
                        df.at[trial_i, 'laser_power'] = 30
                    df.at[trial_i, 'rec_nr'] = rec_nr

                    laser_tick += 1


        if has_pchrome:
            pchr_stim_table = stim_per_mcd_channel['9']
            pchr_tick = 0
            trial_i = trial_offset

            for i, r in pchr_stim_table.iterrows():
                if i == 0:
                    assert r['repeat'] == 1
                    continue

                idx = np.where(r['time'].values > 0)[0]

                if r['repeat'] == 1:
                    assert len(idx) == 1
                    iti = r['time'].iloc[idx[0]] / 1e3
                    df.at[trial_i, 'iti'] = iti
                    trial_i += 1


                if r['repeat'] > 1:
                    # Same logic as with the laser stim params
                    idx = np.where(r['value'].values > 0)[0]
                    assert len(idx) == 1

                    # Now loop over the columns, to see if there is a delay before the first on
                    had_on = False
                    has_delay = False
                    pchr_delay, pchr_on_duration, pchr_off_duration = None, None, None

                    for col_i in range(3):
                        t = r['time'].values[col_i]
                        v = r['value'].values[col_i]

                        if t == 0:  # if time = 0 there is no data in that column
                            continue

                        t = t / 1e3  # convert to [ms]

                        if v == 0:
                            if not had_on:  # if there is a v=0 column before a stimulation, there is an
                                # onset delay
                                pchr_delay = t
                                has_delay = True
                            elif had_on:  # if there is a v=0 column after a stimulation, there is an off
                                # off duration
                                pchr_off_duration = t

                        elif v > 0:
                            assert had_on == False  # redundant, but a value for v can be occuring once in each row
                            pchr_on_duration = t
                            had_on = True

                            if not has_delay:
                                pchr_delay = 0

                    assert had_on


                    df.at[trial_i, 'stimsource'] = rec_info['stimsource']
                    df.at[trial_i, 'rec_nr'] = rec_nr
                    df.at[trial_i, 'pchr_delay'] = pchr_delay
                    df.at[trial_i, 'pchr_on_duration'] = pchr_on_duration
                    df.at[trial_i, 'pchr_off_duration'] = pchr_off_duration
                    df.at[trial_i, 'pchr_repeats'] = r['repeat']

                    df.at[trial_i, 'varied_param'] = rec_info['varied_param']

                    if rec_info['varied_param'] == 'pow':
                        df.at[trial_i, 'pchr_power'] = POLYCHROME_POWERS[pchr_tick]
                    else:
                        df.at[trial_i, 'pchr_power'] = 50

                    pchr_tick += 1

    for i, r in df.iterrows():
        df.at[i, 'train_id'] = f'tid_{i:03d}_{r.rec_nr}'

    if not filepaths.proc_pp_trials.parent.exists():
        filepaths.proc_pp_trials.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepaths.proc_pp_trials)