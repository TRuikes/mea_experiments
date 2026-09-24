from sonogenetics.analysis.lib.data_io import DataIO
import numpy as np
from pathlib import Path
from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis
import utils



dmd_stimfile_dir = Path(r'E:\sono\dmd_stimfiles')

chirp_params = {
    'vec_file': '0_chirp_MEA1_50Hz.vec',
    'bin_file': '0_chirp_MEA1_50Hz.bin',
    'nb_frames_one_chirp': 1600,  # I got this from the 50 Hz chirp file audrey gave
    'fs_chirp': 50,  # [Hz] sample rate of the chirp
    't_pre_chirp_on': 1.5,  # [s] cut the chirp 1.5s before the first on
    'n_bins': 1600, # number of bis per chirp cycle (1 cycle = 32s;
    'nb_chirp_repeats': 20,  # number of times the chirp is repeated in the vec file
}

PADDING_S = 5.082312500000057  # padding in [s] before the vec file. if data is misaligned it could be because
    # this is not correct



def get_chirp_responses(data_io: DataIO,
                        rec_id: str,
                        overwrite=False):

    savename = dataset_dir / 'chirp' / f'{data_io.session_id}-{rec_id}.pkl'
    if savename.exists() and not overwrite:
        return

    stim_info = data_io.burst_df.query(f'rec_id == "{rec_id}"')
    assert len(stim_info) == 1
    stim_info = stim_info.iloc[0]

    vec_path = dmd_stimfile_dir / chirp_params['vec_file']
    vec_keys = np.loadtxt(vec_path)[1:, -1]  # drop header row
    vec_data = np.loadtxt(vec_path)[1:, 1]

    # For me, vec_keys are only 0's
    assert len(np.unique(vec_keys)) == 1

    # Find when and how long the DMD trigger is high (e.g. recording frames)
    dmd_onset_s = stim_info.dmd_burst_onset / 1e3
    dmd_offset_s = stim_info.dmd_burst_offset / 1e3
    dmd_duration_s = dmd_offset_s - dmd_onset_s

    # Find theoretical length of a single chirp
    chirp_duration_s = chirp_params['nb_frames_one_chirp'] / chirp_params['fs_chirp']
    assert dmd_duration_s > chirp_params['nb_chirp_repeats'] * chirp_duration_s

    # padding_s = dmd_duration_s - chirp_params['nb_chirp_repeats'] * chirp_duration_s
    padding_s = PADDING_S

    nb_repeats_measured = int(np.floor(dmd_duration_s / chirp_duration_s))
    assert nb_repeats_measured == 20

    # The vec has a buffer with 0's in front (plot the vec_data)
    # Detect first onset, then step back a little to measure the onset response
    # Time is relative to chirp file (not recorded trigger times)
    true_onset = dmd_onset_s + padding_s
    chirp_on_ms = np.arange(true_onset,
                            true_onset+chirp_params['nb_chirp_repeats']*chirp_duration_s,
                            chirp_duration_s) * 1e3
    chirp_off_ms = chirp_on_ms + chirp_duration_s * 1e3

    # time = np.arange(0, vec_data.size, 1/chirp_params['fs_chirp'])
    # fig = utils.simple_fig()
    # fig.add_scatter(x=time, y=vec_data / np.max(vec_data), showlegend=False)
    # fig.add_scatter(x=[padding_s, padding_s], y=[0,1], showlegend=False)
    # fig.update_xaxes(tickvals=np.arange(0, time[-1], 10), range=[0, 20])
    # utils.save_fig(fig, figure_dir_analysis / 'test.png', display=True)



    # Bin the spikes
    chirp_response_per_cell = {}

    for cid in data_io.cluster_ids:

        # Create output placeholder
        binned_spikes = np.empty((chirp_params['nb_chirp_repeats'], chirp_params['n_bins']))
        sp = data_io.spiketimes[rec_id][cid]
        spikes_per_stim = []
        for repeat_i, (c_start, c_stop) in enumerate(zip(chirp_on_ms, chirp_off_ms)):
            sp_cut = sp[(sp >= c_start) & (sp < c_stop)] - c_start
            binned_spikes[repeat_i, :] = np.histogram(sp_cut, bins=chirp_params['n_bins'],
                                         range=(0, chirp_duration_s*1e3))[0]
            spikes_per_stim.append(sp_cut)

        chirp_response_per_cell[cid] = {
            'spikes_per_stim': spikes_per_stim,
            'binned_spikes': binned_spikes,
            'psth': np.sum(binned_spikes, axis=0),
            'vec_data': vec_data,
            'chirp_onsets_s': chirp_on_ms / 1e3,
            'chirp_duration_s': chirp_duration_s,
            'chirp_padding_s': padding_s,
        }

    savename = dataset_dir / 'chirp' / f'{data_io.session_id}-{rec_id}.pkl'
    utils.save_obj(chirp_response_per_cell, savename)

    print(f'saved: {savename}')



def plot_chirp_responses(data_io: DataIO, rec_id: str):
    loadname = dataset_dir / 'chirp' / f'{data_io.session_id}-{rec_id}.pkl'
    chirp_response_per_cell = utils.load_obj(loadname)
    tasks = []
    for cid, cdata in chirp_response_per_cell.items():
        tasks.append({'chirp_data': cdata, 'session_id': data_io.session_id,
                      'rec_id': rec_id, 'cluster_id': cid})

    savedir = figure_dir_analysis / f'{data_io.session_id}' / f'{rec_id}' / 'chirp'
    if not savedir.exists():
        savedir.mkdir(parents=True)

    utils.run_job(
        job_fn=plot_chirp_response_worker,
        num_threads=10,
        tasks=tasks,
        debug=False,
    )


def plot_chirp_response_worker(chirp_data, session_id, rec_id, cluster_id):
    vec_data = chirp_data['vec_data']
    chirp_onsets_s = chirp_data['chirp_onsets_s']

    padding_i = int(chirp_data['chirp_padding_s'] * chirp_params['fs_chirp'])
    n_idx_chirp = int(chirp_data['chirp_duration_s'] * chirp_params['fs_chirp'])
    single_vec = vec_data[padding_i:padding_i + n_idx_chirp]

    psth = chirp_data['psth']
    sps = chirp_data['spikes_per_stim']

    time = np.arange(0, single_vec.size / chirp_params['fs_chirp'], 1 / chirp_params['fs_chirp'])

    fig = utils.make_figure(
        width=0.6, height=1,
        x_domains={1: [[0.1, 0.9]], 2: [[0.1, 0.9]]},
        y_domains={1: [[0.72, 0.9]], 2: [[0.1, 0.7]]},
    )


    for t in np.arange(0, time[-1], 5):
        fig.add_scatter(x=[t, t], y=[0, 1], mode='lines', line=dict(color='black', width=0.2, dash='1px'),
                        showlegend=False, row=1, col=1, )
        fig.add_scatter(x=[t, t], y=[0, len(sps)], mode='lines', line=dict(color='black', width=0.2, dash='1px'),
                        showlegend=False, row=2, col=1, )

    pos = dict(row=1, col=1)
    fig.add_scatter(
        x=time, y=single_vec / np.max(single_vec),
        mode='lines', line=dict(color='purple', width=0.2, dash='1px'),
        showlegend=False,
        **pos,
    )
    fig.add_scatter(
        x=time, y=psth / np.max(psth),
        mode='lines', line=dict(color='red', width=0.6),
        showlegend=False,
        **pos,
    )

    fig.update_xaxes(
        range=(time[0], time[-1]),
        **pos,
    )


    pos = dict(row=2, col=1)
    x_plot, y_plot = [], []

    for sp_i, sp in enumerate(sps):
        sp_s = sp / 1e3
        x_plot.append(np.vstack([sp_s, sp_s, np.full(sp_s.size, np.nan)]).T.flatten())
        y_plot.append(np.vstack([np.ones(sp_s.size) * sp_i,
                                 np.ones(sp_s.size) * sp_i + 1, np.full(sp_s.size, np.nan)]).T.flatten())

    x_plot = np.hstack(x_plot)
    y_plot = np.hstack(y_plot)

    fig.add_scatter(
        x=x_plot, y=y_plot,
        mode='lines', line=dict(color='black', width=0.4,),
        showlegend=False,
        **pos,
    )
    fig.update_yaxes(
        range=(0, len(sps) + 1),
        **pos,
    )
    fig.update_xaxes(
        range=(time[0], time[-1]),
        tickvals=np.arange(0, time[-1], 10),
        title_text='time [s]',
        **pos,
    )

    savename = figure_dir_analysis / f'{session_id}' / f'{rec_id}' / 'chirp' / f'{cluster_id}.png'
    utils.save_fig(fig, savename, display=False)

