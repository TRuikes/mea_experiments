from BU_hydrogel.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis, data_list
from BU_hydrogel.analysis.lib.data_io import DataIO
from BU_hydrogel.analysis.lib.display_tools import generate_raster_plots_session, generate_heatmaps_session, firing_rate_per_protocol_master, firing_rate_per_param

def main():
    # Setup session ID + create figure output directory

    for session_id in data_list:
        fig_save_dir = figure_dir_analysis / session_id

        if not fig_save_dir.exists():
            fig_save_dir.mkdir(parents=True)

        # Load dataset + dump as pickle to speedup future data loading
        data_io = DataIO(dataset_dir)
        data_io.load_session(session_id, load_pickle=True, load_waveforms=False)
        data_io.dump_as_pickle()

        for i, r in data_io.train_df.iterrows():
            if 'sequence_name' in r.keys():
                data_io.train_df.at[i, 'protocol_name'] = r['sequence_name']
            else:
                data_io.train_df.at[i, 'protocol_name'] = r['recording_name']

        for i, r in data_io.burst_df.iterrows():
            if 'sequence_name' in r.keys():
                data_io.burst_df.at[i, 'protocol_name'] = r['sequence_name']
            else:
                data_io.burst_df.at[i, 'protocol_name'] = r['recording_name']


        generate_raster_plots_session(data_io=data_io, sig_only=True)
        # generate_heatmaps_session(data_io=data_io, sig_only=True)
        # firing_rate_per_protocol_master(data_io=data_io)  # doesn't work for BU

        firing_rate_per_param(data_io, data_io.cluster_ids[5], 99, 99)

        print(f'Finished {session_id}\n\n')


if __name__ == '__main__':
    main()

