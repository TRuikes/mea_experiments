import pandas as pd
import utils
from sonogenetics.analysis.lib.data_io import get_dataio
from sonogenetics.analysis.lib.analysis_tools import detect_preferred_electrode
from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis
from utils import load_obj
from typing import Dict
from sonogenetics.analysis.lib.bootstrap import BootstrapOutput
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


rec_ids = {
# '2026-07-02 mouse c57 650 Mekano6 A': dict(pa='rec_2_A_20260702_pa_intensity_test',
#                                            dmd='rec_5_A_20260702_dmd_full_field_intensities',
#                                            dual= 'rec_4_A_20260702_pa_dmd_timing'),
'2026-09-09 mouse c57 758 Mekano6 A': dict(pa='rec_4_A_20260909_pa_dmd_timing_full_field',
                                           dmd='rec_3_A_20260909_dmd_full_field_intensities',
                                           dual='rec_4_A_20260909_pa_dmd_timing_full_field',
                                           )

}


window = [50, 150]


def main():
    data_io = get_dataio()

    for session_id in rec_ids.keys():
        data_io.load_session(session_id, load_pickle=True)
        # data_io.dump_as_pickle()

        # Load cell response statistics
        load_name = data_io.datadir / f'{data_io.session_id}_cells.csv'
        cells_df = pd.read_csv(load_name, header=[0, 1], index_col=0)

        # get DMD responses
        dmd_rec_id = rec_ids[session_id]['dmd']
        dmd_train_id = data_io.train_df.query(''
                                     f'rec_id == "{dmd_rec_id}" and '
                                     f'dmd_intensity == 250').iloc[0].name
        dmd_responses = get_train_responses(data_io, dmd_train_id)

        # get PA responses
        pa_rec_id = rec_ids[session_id]['pa']
        pa_train_df = data_io.train_df.query(''
                                             f'rec_id == "{pa_rec_id}" and '
                                             f'laser_pulse_repetition_rate == 5000 and '
                                             f'laser_power == 5000 and '
                                             f'has_dmd == False and '
                                             f'has_laser == True and '
                                             f'laser_burst_duration == 100')
        pa_responses = {}
        for train_id, tinfo in pa_train_df.iterrows():
            ec = tinfo.electrode
            pa_responses[ec] = get_train_responses(data_io, train_id)


        # get DUAL responses
        dual_rec_id = rec_ids[session_id]['dual']
        dual_train_df = data_io.train_df.query(''
                                          f'rec_id == "{dual_rec_id}"'
                                          f' and laser_onset_delay == 40'
                                               f' and has_dmd == True'
                                               f' and has_laser == True'
                                               f' and laser_burst_duration == 100')
        dual_responses = {}
        for train_id, tinfo in dual_train_df.iterrows():
            ec = tinfo.electrode
            dual_responses[ec] = get_train_responses(data_io, train_id)

        stats = pd.DataFrame()
        stimsites = []
        for ec, dual_resp_per_ec in dual_responses.items():
            if f'{ec:.0f}' not in stimsites:
                stimsites.append(f'{ec:.0f}')
            for cid, cdata in dual_resp_per_ec.items():

                if cid not in dmd_responses.keys():
                    continue

                dmd_excited = cells_df.loc[cid, (dmd_train_id, 'is_excited')]
                dual_excited = cells_df.loc[cid, (cdata['train_id'], 'is_excited')]

                if cid in pa_responses[ec].keys():
                    pa_excited = cells_df.loc[cid, (pa_responses[ec][cid]['train_id'], 'is_excited')]
                else:
                    pa_excited = 0

                if not dmd_excited:
                    savename = figure_dir_analysis / data_io.session_id / 'response_supression' / f'{ec:.0f}' / 'no_dmd' / cid
                else:
                    savename = figure_dir_analysis / data_io.session_id / 'response_supression' / f'{ec:.0f}' / cid

                count_dmd_dual = get_spike_count_diff(
                    dmd_responses[cid]['bins'],
                    dmd_responses[cid]['binned_sp'],
                    dual_resp_per_ec[cid]['binned_sp'],
                )

                if cid in pa_responses[ec].keys():
                    count_pa_dmd = get_spike_count_diff(
                        dmd_responses[cid]['bins'],
                        dmd_responses[cid]['binned_sp'],
                        pa_responses[ec][cid]['binned_sp'],
                    )
                else:
                    count_pa_dmd = np.nan

                # fig = utils.simple_fig(width=1, height=1,
                #                        subplot_titles={1: [f'DUAL - DMD: {count:.0f}']})
                # fig = add_response_to_fig(fig, dmd_responses[cid], 'rgb(100, 149, 237)', 'rgba(100, 149, 237, 0.3)')
                # fig = add_response_to_fig(fig, resp_per_ec[cid], clr='rgba(147, 112, 219,1)', clr_a='rgba(147, 112, 219,0.3)')
                # if cid in pa_responses[ec].keys():
                #     fig = add_response_to_fig(fig, pa_responses[ec][cid], clr='rgba(200, 20, 20, 1)', clr_a='rgba(200, 20, 20, 0.3)')
                #
                # fig.update_xaxes(tickvals=np.arange(-200, 400, 100), title_text='time [ms]')
                # fig.update_yaxes(title_text='spike count')
                # utils.save_fig(fig, savename, display=False)

                tid = dual_resp_per_ec[cid]['train_id']
                lx, ly = data_io.train_df.loc[tid, 'laser_x'], data_io.train_df.loc[tid, 'laser_y']
                cx, cy = data_io.cluster_df.loc[cid, 'cluster_x'], data_io.cluster_df.loc[cid, 'cluster_y']
                if dmd_excited:
                    stats.at[cid, f'{ec:.0f}_dmd_dual'] = count_dmd_dual

                    if pa_excited:
                        stats.at[cid, f'{ec:.0f}_pa_dmd'] = count_pa_dmd
                    else:
                        stats.at[cid, f'{ec:.0f}_pa_dmd'] = np.nan
                else:
                    stats.at[cid, f'{ec:.0f}_dual'] = np.nan
                    stats.at[cid, f'{ec:.0f}_pa_dmd'] = np.nan

                stats.at[cid, f'{ec:.0f}_d'] = np.sqrt((lx-cx)**2+(ly-cy)**2)

        clrs = ['red', 'green', 'blue']

        # Plot DMD-DUAL contrast
        fig = utils.simple_fig(width=1, height=1)

        for stimsite, clr in zip(stimsites, clrs):
            df_plot = stats[pd.isna(stats[f'{stimsite}_pa_dmd'])]
            y = stats[f'{stimsite}_dmd_dual']
            x = stats[f'{stimsite}_d']
            fig.add_scatter(x=x, y=y, mode='markers', marker=dict(color='blue', size=2), showlegend=False)
            fig.add_scatter(x=[0, np.max(x)], y=[0, 0], mode='lines', line=dict(color='black', width=0.5),
                            showlegend=False)

        fig.update_xaxes(range=[0, np.max(x)], title_text=f'd stimsite - cell [um]',
                         tickvals=np.arange(0, 2000, 500))
        fig.update_yaxes(title_text='\u0394 Spikes', tickvals=np.arange(-1, 1, 0.2))

        savename = figure_dir_analysis / data_io.session_id / 'response_supression' / f'dual-count'
        utils.save_fig(fig, savename, display=True)

        # Plot DMD-DUAL vs DMD-PA map
        fig = utils.simple_fig(width=1, height=1, equal_width_height='y')

        for stimsite, clr in zip(stimsites, clrs):
            y = stats[f'{stimsite}_dmd_dual']
            x = stats[f'{stimsite}_pa_dmd']

            fig.add_scatter(x=x, y=y, mode='markers', marker=dict(color='black', size=2), showlegend=False)
            fig.add_scatter(x=[np.min(x), np.max(x)], y=[0, 0], mode='lines', line=dict(color='black', width=0.5),
                            showlegend=False)


        fig.update_xaxes(title_text=f'\u0394 Spikes (PA - DMD)',
                         tickvals=np.round(np.arange(-1, 1, 0.2), 0))
        fig.update_yaxes(title_text='\u0394 Spikes (DMD - DUAL)',
                         tickvals=np.arange(np.round(-1, 1, 0.2), 0))


        savename = figure_dir_analysis / data_io.session_id / 'response_supression' / f'contrast_dual_dmd_pa'
        utils.save_fig(fig, savename, display=True)


def add_response_to_fig(fig, response_data, clr, clr_a):

    se = np.nanstd(response_data['binned_sp'], axis=0) / np.sqrt(response_data['binned_sp'].shape[0])
    y = np.nanmean(response_data['binned_sp'], axis=0)
    fig.add_scatter(x=response_data['bins'], y=y-se, mode='lines',
                    line=dict(width=0), showlegend=False)
    fig.add_scatter(x=response_data['bins'], y=y+se, mode='lines',
                    line=dict(width=0), showlegend=False, fill='tonexty',
                    fillcolor=clr_a
                    )
    fig.add_scatter(x=response_data['bins'], y=y, mode='lines',
                    line=dict(width=1, color=clr), showlegend=False)
    return fig

def get_spike_count_diff(bins, binned_sp_c0, binned_sp_c1):
    idx = np.where((bins >= window[0]) & (bins < window[1]))[0]
    ns_c1 = np.sum(binned_sp_c1[:, idx])
    ns_c0 = np.sum(binned_sp_c0[:, idx])
    n_max = np.max([ns_c1, ns_c0])
    return (ns_c1 - ns_c0) / n_max


def get_train_responses(data_io, train_id):

    dmd_responses = {}
    for cid in data_io.cluster_ids:
        cluster_data: Dict[str, BootstrapOutput] = load_obj(dataset_dir / 'bootstrapped' / f'bootstrap_{cid}.pkl')
        if cluster_data[train_id] is None:
            continue

        dmd_responses[cid] =  {
            'bins': cluster_data[train_id].get('bins'),
            'fr': cluster_data[train_id].get('firing_rate'),
            'ci_low': cluster_data[train_id].get('firing_rate_ci_low'),
            'ci_high': cluster_data[train_id].get('firing_rate_ci_high'),
            'binned_sp': cluster_data[train_id].get('binned_sp'),
            'train_id': train_id
        }

    return dmd_responses


def plot_DMD_relative_FR(data_io, cells_df):
    # Find cells responding to DMD
    dmd_rec_id = 'rec_5_A_20260702_dmd_full_field_intensities'
    train_df = data_io.train_df.query(f'rec_id == "{dmd_rec_id}"')

    n_cells = len(data_io.cluster_ids)
    response_fr = np.zeros((n_cells, 3))
    is_excited_at_250 = pd.DataFrame()
    for dmd_i, dmd_val in enumerate(['50', '150', '250']):
        tinfo = train_df.query(f'dmd_intensity == {dmd_val}').iloc[0]

        for cluster_i, cid in enumerate(data_io.cluster_ids):

            response_fr[cluster_i, dmd_i] = cells_df.loc[cid, (tinfo.name, 'excitation_max_fr')]

            if dmd_val == '250' and cells_df.loc[cid, (tinfo.name, 'is_excited')]:
                is_excited_at_250.at[cid, 'is_excited'] = 1
            else:
                is_excited_at_250.at[cid, 'is_excited'] = 0

    response_fr = response_fr - np.tile(response_fr[:, 0], [3, 1]).T

    clrs = []
    x_plot = []
    y_plot = []

    for cluster_i in range(n_cells):

        clr = 'black' if is_excited_at_250.loc[data_io.cluster_ids[cluster_i], 'is_excited'] == 0 else 'red'
        x_plot.append([50, 150, 250, np.nan])
        y_plot.append(response_fr[cluster_i, :])
        y_plot.append(np.nan)
        clrs.extend([clr, clr, clr, clr])

    x_plot = np.hstack(x_plot)
    y_plot = np.hstack(y_plot)
    clrs = np.hstack(clrs)

    fig = utils.simple_fig(width=0.5, height=0.6)
    fig.add_scatter(x=x_plot, y=y_plot, mode='lines+markers',
                    marker=dict(color=clrs, size=5),
                    line=dict(color='black', width=1))
    fig.update_yaxes(title_text='relative FR [Hz]', tickvals=np.arange(-100, 100, 20))
    fig.update_xaxes(tickvals=[50, 150, 250], title_text='dmd intensity [a.u.]')
    savename = figure_dir_analysis / data_io.session_id / 'relative_fr_dmd_only'
    if not savename.parent.exists():
        savename.parent.mkdir(parents=True)
    utils.save_fig(fig, savename, display=False)

    return is_excited_at_250











    # # Classify response types
    # laser_onset_delay = 40
    # dmd_onset_delay = 0
    # laser_burst_duration = 20
    # dmd_burst_duration = 20
    #
    # train_df = data_io.train_df.query(
    #     f'laser_onset_delay == {laser_onset_delay} and '
    #     f'dmd_onset_delay == {dmd_onset_delay}  and '
    #     f'rec_id == "rec_4_A_20260702_pa_dmd_timing"'
    # )



if __name__ == '__main__':
    main()