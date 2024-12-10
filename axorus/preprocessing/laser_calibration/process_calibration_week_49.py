#%% Imports and stuff

import pandas as pd
import utils
import numpy as np
from axorus.preprocessing.lib.filepaths import FilePaths
from axorus.preprocessing.project_colors import ProjectColors
from axorus.preprocessing.laser_calibration.oscilloscope_processing import (read_oscilloscope_data,
                                                                        plot_oscilloscope_data, measure_repfreq)
filepaths = FilePaths(laser_calib_week='week_49')
clrs = ProjectColors()

dataframes = []
for f in filepaths.laser_calib_path.iterdir():
    if 'laser_power' not in f.name:
        continue
    # print(f' appending {f.name}')
    dataframes.append(pd.read_csv(f))
power_df = pd.concat(dataframes, ignore_index=True)


#%% Assign some extra columns
for i, r in power_df.iterrows():
    fiber_1, attenuator, fiber_2 = r.fiber_connection.split('_')

    power_df.at[i, 'fiber_1'] = fiber_1
    power_df.at[i, 'attenuator'] = attenuator
    power_df.at[i, 'fiber_2'] = fiber_2

    if fiber_2 ==  'C3':
        diameter = 300 / 1e3  # mm
    elif fiber_2 == 'C7':
        diameter = 100 / 1e3  # mm
    elif fiber_2 == 'C8':
        diameter = 50 / 1e3  # mm
    else:
        raise ValueError()

    power_df.at[i, 'fiber_diameter'] = diameter

    area = np.pi * (diameter / 2) ** 2
    power = r.measured_power / 1000  # W
    irradiance = power / area  # W / mm2

    power_df.at[i, 'irradiance'] = irradiance


#%% Read repeition frequency data

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
        repetition_frequency = 1 / np.mean(dt)

    frep_data.at[i, 'pulse_period'] = pulse_period
    frep_data.at[i, 'repetition_frequency'] = repetition_frequency


# Fit linear curve to frep data
fit_per_laser_level = {}
for laser_level, ldf in frep_data.groupby('ll'):
    x = ldf.dc.values
    y = ldf.repetition_frequency.values

    idx = np.where(pd.notna(y))[0]

    slope, intercept = np.polyfit(x[idx], y[idx], 1)
    fit_per_laser_level[int(laser_level)] = dict(x=x, y=slope*x+intercept, slope=slope, intercept=intercept,
                                            type='measured')

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
                                            type='estimated')

# Write repetition frequencies to power data
for i, r in power_df.iterrows():
    duty_cycle = r.duty_cycle
    laser_level = r.laser_level

    is_estimated = fit_per_laser_level[laser_level]['type'] == 'estimated'
    slope = fit_per_laser_level[laser_level]['slope']
    intercept = fit_per_laser_level[laser_level]['intercept']
    frep = intercept + duty_cycle * slope

    pwr = r.measured_power / 1000  # mW
    epulse = (pwr / frep) * 1e6  # uJ

    if is_estimated:
        power_df.at[i, 'estimated_repetition_frequency'] = frep

        power_df.at[i, 'estimated_energy_pulse'] = epulse
    else:
        power_df.at[i, 'repetition_frequency'] = frep
        power_df.at[i, 'energy_pulse'] = epulse

    power_df.at[i, 'rf_for_plot'] = frep
    power_df.at[i, 'ep_for_plot'] = epulse


#%% Plot power and irradiance per fiber + attenuator

connections_to_plot = (
    ('C3', '26', 'orange'),
    ('C7', '14', 'yellow'),
    ('C8', '04', 'black')
)

fig = utils.make_figure(
    width=0.6, height=1.3,
    x_domains={
        1: [[0.15, 0.9]],
        2: [[0.15, 0.9]],
        3: [[0.15, 0.9]]
    },
    y_domains={
        1: [[0.7, 0.9]],
        2: [[0.45, 0.65]],
        3: [[0.2, 0.4]]
    }
)

# Plot power in the top plot
pos = dict(row=1, col=1)

for fiber_2, attenuator, plot_color in connections_to_plot:

    df = power_df.query(f'fiber_2 == "{fiber_2}" and attenuator == "{attenuator}"'
                        f'and laser_level == 85')
    fiber_diameter = df.iloc[0].fiber_diameter

    x = df.rf_for_plot.values
    y = df.measured_power.values

    fig.add_scatter(
        x=x, y=y,
        mode='lines+markers',
        line=dict(color=plot_color, width=1),
        name=f'{fiber_diameter*1000:.0f} um, {attenuator} mm',
        **pos
    )

fig.update_yaxes(
    tickvals=np.arange(0, 250, 50),
    title_text='P [mW]',
    title_font_size=9,
    **pos,
)


# Plot irradiance in the bottom plot
pos = dict(row=2, col=1)

for fiber_2, attenuator, plot_color in connections_to_plot:

    df = power_df.query(f'fiber_2 == "{fiber_2}" and attenuator == "{attenuator}"'
                        f'and laser_level == 85')

    x = df.rf_for_plot.values
    y = df.irradiance.values

    fig.add_scatter(
        x=x, y=y,
        mode='lines+markers',
        line=dict(color=plot_color, width=1),
        showlegend=False,
        **pos
    )


fig.update_yaxes(
    tickvals=np.arange(0, 30, 5),
    title_text=f'I [W/mm2]',
    title_font_size=9,
    **pos,
)


# Plot the energy per pulse
pos = dict(row=3, col=1)

for fiber_2, attenuator, plot_color in connections_to_plot:

    df = power_df.query(f'fiber_2 == "{fiber_2}" and attenuator == "{attenuator}"'
                        f'and laser_level == 85')

    x = df.rf_for_plot.values
    y = df.ep_for_plot.values

    fig.add_scatter(
        x=x, y=y,
        mode='lines+markers',
        line=dict(color=plot_color, width=1),
        showlegend=False,
        **pos
    )

fig.update_xaxes(
    tickvals=np.arange(0, 2e5, 2000),
    title_text='Pulse rep. f. [Hz]',
    **pos,
)

fig.update_yaxes(
    range=[0, 8],
    tickvals=np.arange(0, 20, 2),
    title_text=f'Pe [uJ]',
    title_font_size=9,
    **pos,
)

utils.save_fig(fig, filepaths.laser_calib_figure_dir / f'power_and_irradiance_per_fiber')


#%% Plot power for continuous attenuator

df = power_df.query('attenuator == "CA"')
fiber_2 = df.iloc[0].fiber_diameter
laser_levels = [85, 95]
for laser_level in laser_levels:

    fig = utils.make_figure(
        width=0.6, height=0.7,
        x_domains={
            1: [[0.1, 0.9]],
            2: [[0.1, 0.9]],
        },
        y_domains={
            1: [[0.6, 0.9]],
            2: [[0.2, 0.5]],
        },
        subplot_titles={
            1: [f'fiber: {fiber_2*1000:.0f} um, laser level: {laser_level}',],
            2: ['']
        }
    )

    pos = dict(row=1, col=1)
    df_ll = df.query(f'laser_level == {laser_level}')
    for dc, dc_df in df_ll.groupby('duty_cycle'):

        x = dc_df.n_turns.values
        y = dc_df.irradiance.values

        rf = dc_df.iloc[0].rf_for_plot

        fig.add_scatter(
            x=x, y=y,
            mode='lines+markers',
            line=dict(color=clrs.duty_cycle(dc), width=1),
            showlegend=True, name=f'Prf: {rf/1000:.0f} kHz',
            **pos,
        )

        # Plot reference line from fixed attenuator
        df_ref = power_df.query('fiber_2 == "C7" and attenuator == "14" and '
                                'duty_cycle == @dc')
        y = [df_ref.iloc[0].irradiance]
        x = [29]

        fig.add_scatter(
            x=[20, 35], y=y * np.ones(2),
            mode='lines',
            line=dict(color=clrs.duty_cycle(dc), width=1, dash='2px'),
            showlegend=False, name=f'Prf: {rf / 1000:.0f} kHz',
            **pos,
        )

    fig.update_xaxes(
        range=[24, 34],
        **pos,
    )


    fig.update_yaxes(
        tickvals=np.arange(0, 35, 1),
        title_text='I [W/mm2]',
        **pos,
    )


    pos = dict(row=2, col=1)
    for dc, dc_df in df_ll.groupby('duty_cycle'):

        x = dc_df.n_turns.values
        y = dc_df.ep_for_plot.values

        rf = dc_df.iloc[0].rf_for_plot

        fig.add_scatter(
            x=x, y=y,
            mode='lines+markers',
            line=dict(color=clrs.duty_cycle(dc), width=1),
            showlegend=False, name=f'Prf: {rf/1000:.0f} kHz',
            **pos,
        )


    fig.update_yaxes(
        tickvals=np.arange(0, 35, 1),
        title_text='Pe [uJ]',
        **pos,
    )


    fig.update_xaxes(
        tickvals=np.arange(21, 35, 2),
        title_text='n turns',
        **pos,
    )

    utils.save_fig(fig, filepaths.laser_calib_figure_dir / f'power_per_number_of_turns_laser_level_{laser_level:.0f}')


