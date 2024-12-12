from pathlib import Path
import pandas as pd
import utils
import numpy as np
from scipy.signal import find_peaks, medfilt
from axorus.preprocessing.lib.filepaths import FilePaths
from axorus.preprocessing.project_colors import ProjectColors
from axorus.preprocessing.laser_calibration.oscilloscope_processing import (read_oscilloscope_data,
                                                                            plot_oscilloscope_data, measure_repfreq)


filepaths = FilePaths(laser_calib_week='week_45')
clrs = ProjectColors()

dataframes = []
for f in filepaths.laser_calib_path.iterdir():
    if 'laser_power' not in f.name:
        continue
    # print(f' appending {f.name}')
    dataframes.append(pd.read_csv(f))
power_df = pd.concat(dataframes, ignore_index=True)

for i, r in power_df.iterrows():

    if 'C1' in r.fiber_connection:
        diameter = 200 / 1e3  # mm
    elif 'C6' in r.fiber_connection:
        diameter = 50 / 1e3  # mm
    elif 'C7' in r.fiber_connection:
        diameter = 50 / 1e3  # mm
    else:
        raise ValueError()

    area = np.pi * (diameter / 2) ** 2  # mm 2
    power = r.measured_power / 1000  # W
    irradiance = power / area  # W / mm2

    power_df.at[i, 'irradiance'] = irradiance  # W / mm2

# Plot power per laser level and fiber connection
for fiber_connection, fdf in power_df.groupby('fiber_connection'):
    fig = utils.simple_fig(
        height=1.5,
        width=1,
        subplot_titles={1: [f'']}
    )

    for laser_level, ldf in fdf.groupby('laser_level'):
        x = ldf.duty_cycle.values
        y = ldf.measured_power.values

        fig.add_scatter(
            x=x,
            y=y,
            mode='lines+markers',
            line=dict(width=1, color=clrs.laser_level(laser_level, 1)),
            marker=dict(size=5, color=clrs.laser_level(laser_level, 1)),
            name=f'L={laser_level}',
        )

    fig.update_xaxes(
        tickvals=power_df.duty_cycle.unique(),
        title_text='Controller duty cycle [no unit]',
    )
    if 'C6' in fiber_connection:
        ymax = 50
        ytick = 10
    else:
        ymax = 50
        ytick = 10

    fig.update_yaxes(
        range=[0, ymax],
        tickvals=np.arange(0, 1200, ytick),
        title_text='Average power [mW]',
    )
    fig.update_layout(
        legend_x=0.9,
        legend_y=0.3,
    )
    # break
    utils.save_fig(fig, filepaths.laser_calib_figure_dir / f'av_pwr_per_laser_level_{fiber_connection}')

# Plot irradiance per cable connection and laser level
for fiber_connection, fdf in power_df.groupby('fiber_connection'):
    if 'C6' in fiber_connection:
        cbl = '50 um'
    elif 'C7' in fiber_connection:
        cbl = '50 um'
    elif 'C1' in fiber_connection:
        cbl = '200 um'
    else:
        raise ValueError()

    att = fiber_connection.split('_')[1]

    fig = utils.simple_fig(
        height=1.5,
        width=1,
        subplot_titles={1: [f'fbr: {cbl}, att: {att}']}
    )

    for laser_level, ldf in fdf.groupby('laser_level'):
        x = ldf.duty_cycle.values
        y = ldf.irradiance.values  # W / mm2

        fig.add_scatter(
            x=x,
            y=y,
            mode='lines+markers',
            line=dict(width=1, color=clrs.laser_level(laser_level, 1)),
            marker=dict(size=5, color=clrs.laser_level(laser_level, 1)),
            name=f'L={laser_level}',
        )

    fig.update_xaxes(
        tickvals=power_df.duty_cycle.unique(),
        title_text='Controller duty cycle [no unit]',
        range=[0, 85]
    )

    if 'C1' in fiber_connection:
        ystep = 5
    elif 'C6' in fiber_connection and att == '14':
        ystep = 5
    elif 'C7' in fiber_connection and att == '14':
        ystep = 5
    elif 'C6' in fiber_connection and att == '04':
        ystep = 5
    else:
        raise ValueError('error?')

    fig.update_yaxes(
        range=[0, 30],
        tickvals=np.arange(0, 1200, ystep),
        title_text='Irradiance [W/mm2]',
    )
    fig.update_layout(
        legend_x=0.9,
        legend_y=0.3,
    )
    # break
    utils.save_fig(fig, filepaths.laser_calib_figure_dir / f'irradiance_per_laser_level_{fiber_connection}')

# Read frep files
frep_data = pd.DataFrame()
file_i = 0
frep_dir = filepaths.laser_calib_path.parent / 'week_41'
for f in (frep_dir / 'repetition_frequency').iterdir():
    if f.is_dir():
        laser_level, duty_cycle = f.name.split('_')
        tick = 0

        for ff in f.iterdir():
            frep_data.at[file_i, 'filename'] = ff.as_posix()
            frep_data.at[file_i, 'dc'] = int(duty_cycle)
            frep_data.at[file_i, 'll'] = int(laser_level)
            frep_data.at[file_i, 'recnr'] = tick

            tick += 1
            file_i += 1

print(f'detected {frep_data.shape[0]} files!')

# detect peaks etc

for i, finfo in frep_data.iterrows():

    f_specs, f_data = read_oscilloscope_data(finfo.filename)

    # Detect peaks
    threshold = 15  # V
    peaks = measure_repfreq(f_data, threshold=threshold)

    time = f_data['x']
    dt = np.diff(time[peaks])
    if dt.size == 0:
        pulse_period = np.nan
        repetition_frequency = np.nan
    else:
        pulse_period = np.mean(dt) * 1e6  # us
        repetition_frequency = 1 / np.mean(dt)  # Hz

    frep_data.at[i, 'pulse_period'] = pulse_period   # us
    frep_data.at[i, 'repetition_frequency'] = repetition_frequency  # Hz

# Fit linear curve to frep data
fit_per_laser_level = {}
for laser_level, ldf in frep_data.groupby('ll'):
    x = ldf.dc.values
    y = ldf.repetition_frequency.values  # Hz

    idx = np.where(pd.notna(y))[0]

    slope, intercept = np.polyfit(x[idx], y[idx], 1)
    fit_per_laser_level[int(laser_level)] = dict(x=x, y=slope*x+intercept, slope=slope, intercept=intercept,
                                            type='measured')  # Hz

measured_levels, measured_slopes, measured_intercepts = [], [], []
for k, v in fit_per_laser_level.items():
    measured_levels.append(k)
    measured_slopes.append(v['slope'])
    measured_intercepts.append(v['intercept'])
measured_levels = np.array(measured_levels)
measured_slopes = np.array(measured_slopes)
measured_intercepts = np.array(measured_intercepts)

fr_slope_slope, fr_slope_intercept = np.polyfit(measured_levels, measured_slopes, 1)
fr_inter_slope, fr_inter_intercept = np.polyfit(measured_levels, measured_intercepts, 1)

for laser_level in power_df.laser_level.unique():
    if laser_level in fit_per_laser_level.keys():
        continue

    fit_slope = fr_slope_intercept + fr_slope_slope * laser_level
    fit_intercept = fr_inter_intercept + fr_inter_slope * laser_level
    x = np.sort(power_df.duty_cycle.unique())
    y = fit_intercept + fit_slope * x
    fit_per_laser_level[int(laser_level)] = dict(x=x, y=y, slope=fit_slope, intercept=fit_intercept,
                                            type='estimated')  # Hz

# Write repetition frequencies to power data
for i, r in power_df.iterrows():
    duty_cycle = r.duty_cycle
    laser_level = r.laser_level

    is_estimated = fit_per_laser_level[laser_level]['type'] == 'estimated'
    slope = fit_per_laser_level[laser_level]['slope']
    intercept = fit_per_laser_level[laser_level]['intercept']
    frep = intercept + duty_cycle * slope

    pwr = r.measured_power / 1000  # convert to [W]
    epulse = (pwr / frep) * 1e6  # [W / Hz * 1e6] = [uJ]

    if is_estimated:
        power_df.at[i, 'estimated_repetition_frequency'] = frep  # Hz

        power_df.at[i, 'estimated_energy_pulse'] = epulse  # [uJ]
    else:
        power_df.at[i, 'repetition_frequency'] = frep  # [Hz]
        power_df.at[i, 'energy_pulse'] = epulse  # [uJ]

# Plot freq rep per laser level
fig = utils.simple_fig(
    height=1.5,
    width=1,
    subplot_titles={1: [f'']}
)

for laser_level, fit_data in fit_per_laser_level.items():
    is_estimated = fit_data['type'] == 'estimated'

    dash = 'solid' if not is_estimated else '1px'
    if not is_estimated:
        name = (f'L={laser_level} (int: {fit_per_laser_level[laser_level]["intercept"]:.0f}, '
                f'slope: {fit_per_laser_level[laser_level]["slope"]:.0f})')
    else:
        name = f'estimated L={laser_level}'

    fig.add_scatter(
        x=fit_data['x'],
        y=fit_data['y'],
        mode='lines',
        line=dict(width=1, color=clrs.laser_level(laser_level, 1), dash=dash),
        marker=dict(size=5, color=clrs.laser_level(laser_level, 1)),
        name=name,
        showlegend=True,
    )

    if not is_estimated:
        ldf = frep_data.query('ll == @laser_level')

        x = ldf.dc.values
        y = ldf.repetition_frequency.values

        fig.add_scatter(
            x=x,
            y=y,
            mode='markers',
            line=dict(width=1, color=clrs.laser_level(laser_level, 1)),
            marker=dict(size=5, color=clrs.laser_level(laser_level, 1),
                        line=dict(width=0.1, color='black')),
            name=f'L={laser_level}',
            showlegend=False,
        )

fig.update_xaxes(
    title_text='Controller duty cycle',
    tickvals=np.arange(0, 100, 20),
)

fig.update_yaxes(
    title_text='Repetition frequency [Hz]',
    tickvals=np.arange(0, 2e4, 1000),
    range=[0, 1.2e4]
)

fig.update_layout(
    legend_x=0.9,
    legend_y=0.3,
)
utils.save_fig(fig, filepaths.laser_calib_figure_dir / f'repetition_frequency_vs_duty_cycle')

# Plot energy per pulse
for fiber_connection, fdf in power_df.groupby('fiber_connection'):
    fig = utils.simple_fig(
        height=1.5,
        width=1,
        subplot_titles={1: [f'']}
    )

    for laser_level, ldf in fdf.groupby('laser_level'):

        x = ldf.duty_cycle.values
        y = ldf.energy_pulse.values

        if not np.any(pd.notna(y)):
            y = ldf.estimated_energy_pulse.values
            is_estimated = True
        else:
            is_estimated = False

        dash = 'solid' if not is_estimated else '1px'

        fig.add_scatter(
            x=x,
            y=y,
            mode='markers+lines',
            line=dict(width=1, color=clrs.laser_level(laser_level, 1),
                      dash=dash),
            marker=dict(size=5, color=clrs.laser_level(laser_level, 1)),
            name=f'L={laser_level}',
            showlegend=False,
        )

    fig.update_xaxes(
        tickvals=power_df.duty_cycle.unique(),
        title_text='Controller duty cycle [no unit]',
    )

    if fiber_connection == 'CB1_26_C1':
        range = [0, 50]
        tickvals = np.arange(0, 60, 10)
    else:
        range = [0, 10]
        tickvals = np.arange(0, 6, 1)

    fig.update_yaxes(
        range=range,
        tickvals=tickvals,
        title_text='Epulse [uJ]',
    )
    fig.update_layout(
        legend_x=0.9,
        legend_y=0.3,
    )
    utils.save_fig(fig, filepaths.laser_calib_figure_dir / f'pulse_energy_{fiber_connection}')

    power_df.to_csv(filepaths.laser_calib_power_file)
    print(f'saved laser calibration: {filepaths.laser_calib_power_file}')

    # To export
    coeffs_fit = pd.DataFrame()

    for fc in power_df.fiber_connection.unique():

        fig = utils.simple_fig(
            height=1.5,
            width=1,
            subplot_titles={1: [f'{fc}']}
        )

        data_to_fit = pd.DataFrame()
        # Find the slope and intercept for each laser level

        for laser_level, ldf in power_df.query('fiber_connection == @fc').groupby('laser_level'):
            idx = np.where(ldf.duty_cycle.values < 90)[0]
            x = ldf.duty_cycle.values[idx]
            y = ldf.measured_power.values[idx]
            idx = np.where(pd.notna(y))[0]

            slope, intercept = np.polyfit(x[idx], y[idx], 1)
            data_to_fit.at[laser_level, 'power_slope'] = slope
            data_to_fit.at[laser_level, 'power_intercept'] = intercept

            fig.add_scatter(
                x=x,
                y=y,
                mode='markers+lines',
                line=dict(width=1, color=clrs.laser_level(laser_level, 1),
                          dash='solid'),
                marker=dict(size=5, color=clrs.laser_level(laser_level, 1)),
                name=f'L={laser_level}',
                showlegend=False,
            )

        # Power is in [mW]
        coeffs_fit.at[fc, 'power_slope'] = data_to_fit.loc[85].power_slope   # mW
        coeffs_fit.at[fc, 'power_intercept'] = data_to_fit.loc[85].power_intercept  # mW
        coeffs_fit.at[fc, 'fr_slope_slope'] = fr_slope_slope
        coeffs_fit.at[fc, 'fr_slope_intercept'] = fr_slope_intercept
        coeffs_fit.at[fc, 'fr_inter_slope'] = fr_inter_slope
        coeffs_fit.at[fc, 'fr_inter_intercept'] = fr_inter_intercept

        for l_test in [85]:
            s = coeffs_fit.loc[fc, 'power_slope']
            i = coeffs_fit.loc[fc, 'power_intercept']

            x = np.arange(0, 100, 20)
            y = s * x + i
            fig.add_scatter(
                x=x,
                y=y,
                mode='markers+lines',
                line=dict(width=1, color=clrs.laser_level(l_test, 1),
                          dash='1px'),
                marker=dict(size=5, color=clrs.laser_level(l_test, 1)),
                name=f'L={l_test}',
                showlegend=False,
            )

        fig.update_xaxes(
            tickvals=power_df.duty_cycle.unique(),
            title_text='Controller duty cycle [no unit]',
        )

        if fc == 'CB1_26_C1':
            ymax = 500
        else:
            ymax = 50

        fig.update_yaxes(
            range=[0, ymax],
            tickvals=np.arange(0, 1200, ymax/10),
            title_text='Average power [mW]',
        )

        utils.save_fig(fig, filepaths.laser_calib_figure_dir / f'fit_power_curves_{fc}')

coeffs_fit.to_csv(filepaths.laser_calib_file)
print(f'saved: {filepaths.laser_calib_file}')
# measured_power as function of dc, per laser level
# repetition frequency as function of dc, per laser level
