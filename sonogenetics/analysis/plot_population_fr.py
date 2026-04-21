from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis, data_list
from sonogenetics.analysis.lib.data_io import DataIO
import pandas as pd
from sonogenetics.analysis.lib.analysis_tools import detect_preferred_electrode
from utils import  make_figure, save_fig
import numpy as np

colors = {
    "none": "lightgray",
    "in": "#1f77b4",  # blue
    "ex": "#d62728",  # red
    "ex_in": "#9467bd"  # purple
}



def plot_firing_rate(data_io: DataIO):
    loadname = dataset_dir / f'{data_io.session_id}_cells.csv'
    cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)
    pref_ec = detect_preferred_electrode(data_io, cells_df)

    burst_duration = 20
    laser_prr = 4000

    assert burst_duration in data_io.train_df.laser_burst_duration.values
    assert laser_prr in data_io.train_df.laser_pulse_repetition_rate.values

    for rec_id, rdict in pref_ec.items():
        for protocol, pdf in rdict.items():

            if 'dmd' in protocol:
                continue

            # Make figure
            date, species, strain, animal_id, channel, slice_nr = data_io.session_id.split()
            fig = make_figure(
                width=0.5,
                height=0.8,
                subplot_titles={1: [f'{channel} {animal_id}']}
            )

            cell_cat_df = pd.DataFrame()

            for cluster_id, cinfo in pdf.iterrows():

                if pd.isna(cinfo.ec):
                    continue

                trials = data_io.train_df.query(
                    f'protocol == "{protocol}" and '
                    f'rec_id == "{rec_id}" and '
                    f'electrode == {cinfo.ec} and '
                    f'laser_burst_duration == {burst_duration} and '
                    f'laser_pulse_repetition_rate == {laser_prr}'
                )

                n_trials = len(trials)

                cell_bs_fr = np.zeros(n_trials)
                cell_fr = np.zeros(n_trials)
                cell_prr = np.zeros(n_trials)
                cell_pwr = np.zeros(n_trials)
                cell_clr = []

                train_i = 0
                for train_id, tinfo in trials.iterrows():
                    cell_prr[train_i] = tinfo.laser_pulse_repetition_rate
                    if 'dac_voltage' not in tinfo.keys():
                        pwr = tinfo.laser_power
                    else:
                        pwr = tinfo.dac_voltage

                    cell_pwr[train_i] = pwr
                    cell_bs_fr[train_i] = cells_df.loc[cluster_id, train_id]['baseline_firing_rate_max']

                    is_excited = cells_df.loc[cluster_id, train_id]['is_excited']
                    is_inhibited = cells_df.loc[cluster_id, train_id]['is_inhibited']
                    if is_excited:

                        cell_fr[train_i] = cells_df.loc[cluster_id, train_id]['excitation_max_fr']
                        cell_clr.append('red')

                    elif is_inhibited:
                        cell_fr[train_i] = cells_df.loc[cluster_id, train_id]['inhibition_min_fr']
                        cell_clr.append('blue')
                    else:
                        cell_fr[train_i] = cells_df.loc[cluster_id, train_id]['max_response_if_not_sig']
                        cell_clr.append('black')

                    if is_excited and is_inhibited:
                        cat = 'ex_in'
                    elif is_excited and not is_inhibited:
                        cat = 'ex'
                    elif not is_excited and is_inhibited:
                        cat = 'in'
                    else:
                        cat = 'none'

                    cell_cat_df.at[cluster_id, pwr] = cat

                    train_i += 1

                cell_fr = cell_fr - cell_bs_fr
                cell_bs_fr = cell_bs_fr - cell_bs_fr
                # cell_bs_fr = cell_bs_fr / fr_max
                # cell_fr = cell_fr / fr_max

                plot_x = []
                plot_y = []
                plot_clr = []
                for i in range(n_trials):
                    jitter = np.random.randint(-50, 50, 1)[0]
                    plot_x.extend([cell_pwr[i] - 250 + jitter, cell_pwr[i] + jitter, None])
                    plot_y.extend([cell_bs_fr[i], cell_fr[i], None])
                    plot_clr.extend(['black', cell_clr[i], 'black'])

                plot_x = np.array(plot_x)
                plot_y = np.array(plot_y)

                fig.add_scatter(
                    x=plot_x,
                    y=plot_y,
                    mode='lines+markers',
                    line=dict( width=0.5, color='grey'),
                    marker=dict(size=5, color=plot_clr),
                    showlegend=False,
                )

            trials = data_io.train_df.query(
                f'protocol == "{protocol}" and '
                f'rec_id == "{rec_id}" and '
                f'laser_burst_duration == {burst_duration}'
            )
            pwrs = cell_cat_df.columns


            fig.update_xaxes(
                title_text="Laser power [controller setting]",
                tickvals=pwrs,
                ticktext=pwrs,
            )

            fig.update_yaxes(
                title_text="\u0394 FR. [Hz]",
                tickvals=np.arange(-100, 200, 50),
            )
            savename = figure_dir_analysis / 'population_firing_rates' / f'{animal_id}_{channel}_{rec_id}_{protocol}_firing_rates'
            save_fig(fig=fig, savename=savename, display=False)

            pwrs = np.sort(cell_cat_df.columns)
            n_pwrs = cell_cat_df.shape[1]
            if n_pwrs == 0:
                print('hi')
            x_dom = []
            y_dom = []
            x_spacing = 0.05
            x_offset = 0.1
            x_width = (1 - 2 * x_offset - (n_pwrs-1) * x_spacing) / n_pwrs

            for i in range(n_pwrs):
                x0 = x_offset + i * (x_width + x_spacing)
                x_dom.append([x0, x0+x_width])
                y_dom.append([0.1, 0.9])

            fig_pie = make_figure(
                width=1,
                height=1,
                x_domains={1: x_dom},
                y_domains={1: y_dom},
                specs=[[{'type': 'domain'} for _ in range(n_pwrs)]],
                subplot_titles={1: [f'power: {pwr:.0f}' for pwr in pwrs]}
            )
            order = ['ex_in', 'ex', 'in', 'none']
            colors  =['#C77DFF', '#FF6B6B', '#4CC9F0', '#111827']

            for i, pwr in enumerate(pwrs):
                counts = cell_cat_df[pwr].value_counts().reindex(order, fill_value=0)

                fig_pie.add_pie(
                    values=counts.values,
                    labels=counts.index,
                    marker=dict(colors=colors),
                    textinfo='label+percent',  # show labels on slices
                    textposition='inside',
                    direction="clockwise",
                    pull=[0.01, 0.01, 0.01, 0],
                    sort=False,
                    row=1,
                    col=i+1,

                )

            fig_pie.update_layout(
                showlegend=False,
            )
            savename = figure_dir_analysis / 'population_firing_rates' / f'{animal_id}_{channel}_{rec_id}_{protocol}_pie'
            save_fig(fig=fig_pie, savename=savename, display=False)



def get_stats(data_io: DataIO):
    loadname = dataset_dir / f'{data_io.session_id}_cells.csv'
    cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)

    date, species, strain, animal_id, channel, slice_nr = data_io.session_id.split()

    burst_duration = 20
    laser_prr = 4000

    assert burst_duration in data_io.train_df.laser_burst_duration.values
    assert laser_prr in data_io.train_df.laser_pulse_repetition_rate.values

    cell_cat_df = pd.DataFrame()
    row_i = 0

    for rec_id in data_io.recording_ids:
        for (protocol, electrode), pdf in data_io.train_df.query(f'rec_id == "{rec_id}"').groupby(['protocol', 'electrode']):
            if 'dmd' in protocol:
                continue


            for cluster_id in data_io.cluster_ids:

                if cluster_id == 'uid_2026-03-25 mouse c57 617 Mekano6 B_005':
                    print( 'hi')

                trials = pdf.query(
                    f'protocol == "{protocol}" and '
                    f'rec_id == "{rec_id}" and '
                    f'electrode == {electrode} and '
                    f'laser_burst_duration == {burst_duration} and '
                    f'laser_pulse_repetition_rate == {laser_prr}'
                )

                for train_id, tinfo in trials.iterrows():
                    if 'dac_voltage' not in tinfo.keys():
                        pwr = tinfo.laser_power
                    else:
                        pwr = tinfo.dac_voltage

                    is_excited = cells_df.loc[cluster_id, train_id]['is_excited']
                    is_inhibited = cells_df.loc[cluster_id, train_id]['is_inhibited']

                    if pd.isna(is_excited) or pd.isna(is_inhibited):
                        continue

                    if isinstance(is_excited, np.bool):
                        is_excited = True if is_excited == np.True_ else False

                    if isinstance(is_inhibited, np.bool):
                        is_inhibited = True if is_inhibited == np.True_ else False

                    assert isinstance(is_excited, bool)
                    assert isinstance(is_inhibited, bool)


                    if is_excited is True and is_inhibited is True:
                        cat = 'ex_in'
                    elif is_excited is True and is_inhibited is False:
                        cat = 'ex'
                    elif is_excited is False and is_inhibited is True:
                        cat = 'in'
                    else:
                        cat = 'none'

                    cx = data_io.cluster_df.loc[cluster_id, 'cluster_x']
                    cy = data_io.cluster_df.loc[cluster_id, 'cluster_y']
                    lx = data_io.train_df.loc[train_id, 'laser_x']
                    ly = data_io.train_df.loc[train_id, 'laser_y']

                    d = np.sqrt((lx - cx)**2 + (ly - cy)**2)


                    cell_cat_df.at[row_i, 'cluster_id'] = cluster_id
                    cell_cat_df.at[row_i, 'train_id'] = train_id
                    cell_cat_df.at[row_i, 'pwr'] = pwr
                    cell_cat_df.at[row_i, 'cat'] = cat
                    cell_cat_df.at[row_i, 'rec_id'] = rec_id
                    cell_cat_df.at[row_i, 'protocol'] = protocol
                    cell_cat_df.at[row_i, 'electrode'] = electrode
                    cell_cat_df.at[row_i, 'd'] = d
                    cell_cat_df.at[row_i, 'channel'] = channel

                    if is_excited:
                        cell_cat_df.at[row_i, 'latency'] = cells_df.loc[cluster_id, (train_id, 'excitation_start')]
                        cell_cat_df.at[row_i, 'fr'] = cells_df.loc[cluster_id, (train_id, 'excitation_max_fr')]

                    elif is_inhibited:
                        cell_cat_df.at[row_i, 'latency'] = cells_df.loc[cluster_id, (train_id, 'inhibition_start')]


                    row_i += 1

    return cell_cat_df


def plot_frac_responding_per_power(cell_cat_df):

    for rec_id in cell_cat_df.rec_id.unique():

        df = cell_cat_df.query('rec_id == @rec_id and d < 250')

        df_stats = pd.DataFrame()
        for (pwr, cat), _ in df.groupby(['pwr', 'cat']):
            df_stats.at[pwr, cat] = 0

        for stimsite, sdf in df.groupby('electrode'):
            n_cells = sdf.cluster_id.nunique()

            print(f'n_cells: {n_cells}')
            for pwr, pdf in sdf.groupby('pwr'):
                txt = f'{pwr}\t'
                for cat, cnt in pdf.cat.value_counts().items():
                    txt += f'{cat} {cnt} | '

                    df_stats.loc[pwr, cat] += cnt

                print(txt)

        # Fill NaNs with 0
        df_stats = df_stats.fillna(0)

        # Convert to fractions (row-wise normalization)
        df_frac = df_stats.div(df_stats.sum(axis=1), axis=0)

        order = ["none", "in", "ex", "ex_in"]
        order = [o for o in order if o in df_frac.columns]
        df_frac = df_frac[order]

        fig = make_figure(
            width=0.7,
            height=0.6,
        )

        for cat in order:
            fig.add_bar(
                x=df_frac.index.astype(str),
                y=df_frac[cat],
                name=cat,
                marker_color=colors[cat]
            )

        fig.update_layout(
            barmode='stack',
        )

        channel = df.iloc[0].channel

        fig.update_xaxes(
            tickmode='array',
            tickvals=[4000, 5000, 6000, 7000],
            ticktext=["4", "5", "6", "7"],  # or keep as strings if you prefer
            title_text=f'{channel} - Laser power',
        )

        fig.update_yaxes(
            title_text='% cells',
            ticklen=3,
            tickvals=np.arange(0, 1.5, 0.25)
        )

        savename = figure_dir_analysis / 'barplots_frac_cells' / f'{channel}_{rec_id}_pie'
        save_fig(fig=fig, savename=savename, display=True)


def plot_response_latency(cell_cat_df):


    for rec_id in cell_cat_df.rec_id.unique():

        df = cell_cat_df.query('rec_id == @rec_id and d < 250 and pwr == 6000')

        fig = make_figure(
            width=0.3,
            height=1,
            x_domains={1: [[0.3, 0.95]]}
        )

        for cat, cat_df in df.groupby('cat'):
            if cat == 'none':
                continue

            y = cat_df.latency.values

            print(rec_id, cat, y.size)

            fig.add_violin(
                x=np.zeros_like(y),
                y=y,
                marker=dict(color=colors[cat]),
                points='all',
                spanmode='hard',
                width=0.3,
                showlegend=False,
            )

        fig.update_yaxes(
            title_text=f'Latency (ms)',
            tickvals=np.arange(0, 200, 50),
            range=[0, 200,],
        )

        channel = df.iloc[0].channel

        fig.update_xaxes(
            tickvals=[0],
            ticktext=[channel]
        )

        savename = figure_dir_analysis / 'latencies' / f'{channel}_{rec_id}_pie'
        save_fig(fig=fig, savename=savename, display=True)


def plot_firing_rates(cell_cat_df):
    for rec_id in cell_cat_df.rec_id.unique():
        df = cell_cat_df.query('rec_id == @rec_id and d < 250 and cat == "ex"')

        fig = make_figure(
            width=0.3,
            height=1,
            x_domains={1: [[0.3, 0.95]]}
        )

        for cid, cdf in df.groupby('cluster_id'):

            y = pwr_df.fr.values

            fig.add_violin(
                x=np.ones_like(y) * pwr,
                y=y,
                marker=dict(color='black'),
                points='all',
                spanmode='hard',
                width=0.3,
                showlegend=False,
            )

        fig.update_yaxes(
            title_text=f'Fr (Hz)',
            tickvals=np.arange(0, 200, 50),
            range=[0, 200, ],
        )

        channel = df.iloc[0].channel

        fig.update_xaxes(
            tickvals=[0],
            ticktext=[channel]
        )

        savename = figure_dir_analysis / 'latencies' / f'{channel}_{rec_id}_pie'
        save_fig(fig=fig, savename=savename, display=True)


def main():
    # Setup session ID + create figure output directory

    for session_id in data_list:
        fig_save_dir = figure_dir_analysis / session_id

        if not fig_save_dir.exists():
            fig_save_dir.mkdir(parents=True)

        # Load dataset + dump as pickle to speedup future data loading
        data_io = DataIO(dataset_dir)
        data_io.load_session(session_id, load_pickle=True, load_waveforms=False)

        for i, r in data_io.train_df.iterrows():
            if 'sequence_name' in r.keys():
                data_io.train_df.at[i, 'protocol'] = r['sequence_name']
            else:
                data_io.train_df.at[i, 'protocol'] = r['recording_name']

        for i, r in data_io.burst_df.iterrows():
            if 'sequence_name' in r.keys():
                data_io.burst_df.at[i, 'protocol'] = r['sequence_name']
            else:
                data_io.burst_df.at[i, 'protocol'] = r['recording_name']

        cell_cat_df = get_stats(data_io)

        plot_frac_responding_per_power(cell_cat_df)
        # plot_response_latency(cell_cat_df)

        print(f'Finished {session_id}\n\n')


if __name__ == '__main__':
    main()

