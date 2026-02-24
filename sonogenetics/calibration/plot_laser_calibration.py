import sys
sys.path.append('.')
import utils

# from utils import *
from pathlib import Path
import json
import pandas as pd
from sonogenetics.project_colors import ProjectColors
import numpy as np

CALIBRATION_PATH = r'C:\sono'  # This path contains all the laser calibration files


def main():
    laser_calibration_files = [f for f in Path(CALIBRATION_PATH).iterdir() if 'laser_calibration' in f.name]
    laser_calibration_dates = [f.name.split('_')[0] for f in laser_calibration_files]

    for f, d in zip(laser_calibration_files, laser_calibration_dates):
        print(f)
        data = read_calibration_file(f)
        plot_calibration_data(data, d)


def read_calibration_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame()
    i = 0
    for fiber_connection, fiber_data in data.items():
        # metadata = fiber_data['metadata']
        
        for prr, prr_data in fiber_data.items():

            if prr == 'metadata':
                continue

            for power, power_data in prr_data.items():
                df.at[i, 'fiber_connection'] = fiber_connection
                df.at[i, 'pulse_repetition_rate'] = float(prr)
                df.at[i, 'voltage'] = float(power)
                df.at[i, 'power'] = float(power_data)
                # df.at[i, 'burst_duration'] = metadata['burst_duration']
                # df.at[i, 'burst_period'] = metadata['burst_period']
                # df.at[i, 'pulsewidth'] = metadata['pulsewidth']
                i += 1
    return df


def plot_calibration_data(data: pd.DataFrame, rec_date):

    clrs = ProjectColors()

    connections = [c for c in data.fiber_connection.unique() if c != 'none']
    for c in connections:
        fig = utils.make_figure(
            width    = 1,
            height   = 1,
            x_domains={1: [[0.1, 0.9]],
                                    
                    },
            y_domains={1: [[0.1, 0.9]], 
                   },
            subplot_titles={1: [''], }
        )

        # fig = go.Figure()

        xtext, xticks = None, None

        ymin, ymax = None, None

        c_df = data.query('fiber_connection == @c')
        for prr, prr_df in c_df.groupby('pulse_repetition_rate'):
            x = prr_df['voltage'].values
            y = prr_df['power'].values
            clr = clrs.repetition_frequency(int(prr))

            fig.add_scatter(
                x=x, y=y, mode='markers+lines',
                line=dict(color=clr),
                marker=dict(color=clr),
                showlegend=True,
                name=f'prr={prr}Hz',
            )

            if xtext is None:
                xtext = prr_df['voltage'].values
                xticks = prr_df['voltage'].values   

            if ymin is None or prr_df['power'].min() < ymin:
                ymin = prr_df['power'].min()
            if ymax is None or prr_df['power'].max() > ymax:
                ymax = prr_df['power'].max()

            

        fig.update_xaxes(
            tickvals=np.arange(3000, 6001, 250),
            ticktext=np.arange(3000, 6001, 250),
            title_text='DAC voltage [V]',
            row=1, col=1,
        )

        yticks = np.round(np.arange(ymin, ymax+1, (ymax-ymin)/5), -1)

        fig.update_yaxes(
            tickvals=yticks,
            ticktext=yticks,
            title_text='Power [mW]',
            range=[ymin, ymax*1.1],
            row=1, col=1
        )

        savename = Path(CALIBRATION_PATH) / 'calibration_figures' / f'{rec_date}_{c}'
        utils.save_fig(fig, savename, display=True if 'c2' in c else False)


if __name__ == '__main__':
    main()