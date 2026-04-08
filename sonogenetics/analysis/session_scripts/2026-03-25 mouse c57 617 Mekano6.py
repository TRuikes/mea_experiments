import os
import pandas as pd
from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis
from sonogenetics.analysis.lib.data_io import DataIO
from sonogenetics.analysis.lib.display_tools import generate_raster_plots_session, generate_heatmaps_session

def main():
    # Setup session ID + create figure output directory
    session_id = '2026-03-25 mouse c57 617 Mekano6 B'
    fig_save_dir = figure_dir_analysis / session_id
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)

    # Load dataset + dump as pickle to speedup future data loading
    data_io = DataIO(dataset_dir)
    data_io.load_session(session_id, load_pickle=True, load_waveforms=False)
    data_io.dump_as_pickle()

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


    generate_raster_plots_session(data_io=data_io)
    generate_heatmaps_session(data_io=data_io, sig_only=True)

if __name__ == '__main__':
    main()
