from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis
from sonogenetics.analysis.data_list import data_list
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

MIN_D = 300
LASER_POWER = 5000
LASER_PRR = 5000
LASER_BURST_DURATION = 20


def get_stats(data_io: DataIO):
    loadname = dataset_dir / f'{data_io.session_id}_cells.csv'
    cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)

    date, species, strain, animal_id, channel, slice_nr = data_io.session_id.split()

    assert LASER_BURST_DURATION in data_io.train_df.laser_burst_duration.values
    assert LASER_PRR in data_io.train_df.laser_pulse_repetition_rate.values

    pref_ec = detect_preferred_electrode(data_io, cells_df)

    cell_cat_df = pd.DataFrame()
    row_i = 0

    for (rec_id, protocol, electrode), rdf in data_io.train_df.groupby(['rec_id', 'protocol_name', 'electrode']):
        if 'dmd' in protocol:
            continue

        for cluster_id in data_io.cluster_ids:
            if rec_id not in pref_ec.keys():
                continue

            if electrode != pref_ec[rec_id][protocol].loc[cluster_id, 'ec']:
                continue

            trials = rdf.query(
                f'laser_burst_duration == {LASER_BURST_DURATION} and '
                f'laser_pulse_repetition_rate == {LASER_PRR}'
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
                    cell_cat_df.at[row_i, 'fr'] = cells_df.loc[cluster_id, (train_id, 'inhibition_min_fr')]

                row_i += 1

    return cell_cat_df

def plot_frac_responding_per_power(cell_cat_df, data_io):

    for rec_id in cell_cat_df.rec_id.unique():

        df = cell_cat_df.query(f'rec_id == @rec_id and d < {MIN_D}')

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

        xticks = []

        for cat in order:
            for xt in df_frac.index.values:
                if xt not in xticks:
                    xticks.append(xt)
            fig.add_bar(
                x=df_frac.index.values,
                y=df_frac[cat],
                name=cat,
                marker_color=colors[cat]
            )

        fig.update_layout(
            barmode='stack',
        )

        channel = df.iloc[0].channel

        xticks = np.sort(xticks)
        fig.update_xaxes(
            tickmode='array',
            tickvals=[f'{x}' for x in xticks],
            ticktext=[f'{x/1000:.1f}' for x in xticks],  # or keep as strings if you prefer
            title_text=f'{channel} - Laser power',
        )

        fig.update_yaxes(
            title_text='% cells',
            ticklen=3,
            tickvals=np.arange(0, 1.5, 0.25)
        )

        savename = figure_dir_analysis / data_io.session_id / 'barplots_frac_cells' / f'{channel}_{rec_id}_pie'
        save_fig(fig=fig, savename=savename, display=True)

def plot_firing_rate(cell_cat_df: pd.DataFrame, data_io):
    df = cell_cat_df.query(f'd < {MIN_D}')

    channel = df.iloc[0].channel
    for (rec_id, protocol), rdf in df.groupby(['rec_id', 'protocol']):

        fig = make_figure(
            width=0.7,
            height=0.6,
        )

        pwrs = np.sort(rdf.pwr.unique())
        cats = ['ex', 'in', 'none']

        xticks = []
        xvals = []

        for pi, pwr in enumerate(pwrs):
            xticks.append(pi * 4 +1)
            xvals.append(pwr / 1000)

            for ci, cat in enumerate(cats):
                x = pi * 4 + ci
                df = rdf.query(f'pwr == {pwr} and cat == "{cat}"')
                y = df.fr.values



                fig.add_violin(
                    x=np.ones_like(y) * x,
                    y=y,
                    marker=dict(color=colors[cat], size=3),
                    points='all',
                    spanmode='hard',
                    width=1,
                    showlegend=False,
                )

        fig.update_xaxes(
            tickvals=xticks,
            ticktext=[f'{p:.1f}' for p in xvals],
            title_text=f'{channel} - Laser power',
        )

        fig.update_yaxes(
            title_text=f'Fr [Hz]',
            tickvals=np.arange(0, 300, 50),
            range=[0, 300]
        )
        savename = figure_dir_analysis / data_io.session_id / 'firing_rate_cells' / f'{channel}_{rec_id}'
        save_fig(fig=fig, savename=savename, display=True)


def plot_response_latency(cell_cat_df, data_io):
    df = cell_cat_df.query(f'd < {MIN_D}')

    for (rec_id, protocol), rdf in df.groupby(['rec_id', 'protocol']):



        fig = make_figure(
            width=0.7,
            height=0.6,
        )


        pwrs = np.sort(rdf.pwr.unique())
        cats = ['ex', 'in', 'none']

        xticks = []
        xvals = []

        for pi, pwr in enumerate(pwrs):
            xticks.append(pi * 4 +1)
            xvals.append(pwr / 1000)

            for ci, cat in enumerate(cats):
                x = pi * 4 + ci
                df = rdf.query(f'pwr == {pwr} and cat == "{cat}"')
                y = df.latency.values

                fig.add_violin(
                    x=np.ones_like(y) * x,
                    y=y,
                    marker=dict(color=colors[cat], size=3),
                    points='all',
                    spanmode='hard',
                    width=1,
                    showlegend=False,
                )


        fig.update_yaxes(
            title_text=f'Latency (ms)',
            tickvals=np.arange(0, 100, 25),
            range=[0, 100,],
        )

        # if len(df) == 0:
        #     continue
        channel = df.iloc[0].channel

        fig.update_xaxes(
            tickvals=xticks,
            ticktext=xvals
        )

        savename = figure_dir_analysis / data_io.session_id / 'latencies' / f'{channel}_{rec_id}_pie'
        save_fig(fig=fig, savename=savename, display=True)


def plot_mean_firing_rate(cell_cat_df, data_io):

    channel = cell_cat_df.iloc[0].channel
    for (rec_id, protocol), rdf in cell_cat_df.groupby(['rec_id', 'protocol']):

        fig = make_figure(
            width=0.7,
            height=0.6,
        )

        pwrs = np.sort(rdf.pwr.unique())
        cats = ['ex', 'in', 'none']

        for ci, cat in enumerate(cats):
            x_plot = []
            y_plot = []
            y_se_plot = []
            for pi, pwr in enumerate(pwrs):
                x = pi * 4 + ci
                df = rdf.query(f'cat == "{cat}" and pwr == {pwr}')
                y = np.mean(df.fr.values)
                y_se = np.std(df.fr.values) / np.sqrt(len(df.fr.values))
                x_plot.append(x)
                y_plot.append(y)
                y_se_plot.append(y_se)

            fig.add_scatter(
                x=x_plot,
                y=y_plot,
                error_y=dict(array=y_se_plot),
                marker=dict(color=colors[cat]),
                showlegend=False,
            )

        fig.update_xaxes(
            tickvals=[i*3 + 2 for i in range(len(pwrs))],
            ticktext=[f'{p/1000:.1f}' for p in pwrs],
            title_text=f'{channel} - Laser power',
        )

        fig.update_yaxes(
            title_text=f'Fr [Hz]',
            tickvals=np.arange(0, 200, 50),
        )
        savename = figure_dir_analysis / data_io.session_id / 'firing_rate_cells_mean' / f'{channel}_{rec_id}'
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

        cell_cat_df = get_stats(data_io)

        plot_frac_responding_per_power(cell_cat_df, data_io)
        plot_firing_rate(cell_cat_df, data_io)
        plot_mean_firing_rate(cell_cat_df, data_io)
        plot_response_latency(cell_cat_df, data_io)

        print(f'Finished {session_id}\n\n')


if __name__ == '__main__':
    main()

