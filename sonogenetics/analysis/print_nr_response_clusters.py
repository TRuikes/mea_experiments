import os
import pandas as pd
from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis, data_list
from sonogenetics.analysis.lib.data_io import DataIO
from sonogenetics.analysis.lib.analysis_tools import detect_preferred_electrode, get_params_protocol, params_abbreviation


def main():

    for session_id in data_list:
        print(f'\n\nSession: {session_id}')
        fig_save_dir = figure_dir_analysis / session_id
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)

        # Load dataset + dump as pickle to speedup future data loading
        data_io = DataIO(dataset_dir)
        data_io.load_session(session_id, load_pickle=True, load_waveforms=False)
        data_io.dump_as_pickle()

        loadname = dataset_dir / f'{data_io.session_id}_cells.csv'
        cells_df = pd.read_csv(loadname, header=[0, 1], index_col=0)

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

        pref_ec = detect_preferred_electrode(data_io, cells_df)



        for (rec_id, protocol), rdf in data_io.train_df.groupby(['rec_id', 'protocol']):

            print(f'\t{rec_id} - {protocol}')
            results = pd.DataFrame()
            i = 0

            for cluster_id in cells_df.index.tolist():
                for ec, edf in rdf.groupby('electrode'):

                    n_sig = 0
                    for tid in edf.index.values:

                        is_excited = cells_df.loc[cluster_id, (tid, 'is_excited')]
                        is_inhibited = cells_df.loc[cluster_id, (tid, 'is_inhibited')]
                        results.loc[i, 'cluster_id'] = cluster_id
                        results.loc[i, 'ec'] = ec
                        results.loc[i, 'tid'] = tid

                        if is_excited is True and is_inhibited is True:
                            cat = 'ex_in'
                            n_sig += 1
                        elif is_excited is True and is_inhibited is False:
                            cat = 'ex'
                            n_sig += 1

                        elif is_excited is False and is_inhibited is True:
                            cat = 'in'
                            n_sig += 1

                        else:
                            cat = 'none'


                        results.loc[i, 'cat'] = cat

                        i += 1

            cluster_labels = (
                results.groupby("cluster_id")["cat"]
                .agg(lambda x: x.value_counts().idxmax())
            )
            cluster_labels = cluster_labels.reset_index()
            cluster_labels.columns = ["cluster_id", "final_cat"]

            category_counts = cluster_labels["final_cat"].value_counts()

            for i, k in category_counts.items():
                print(f'\t\t{i} - {k}')



            # non_none_counts = (
            #     results[results["cat"] != "none"]
            #     .groupby("cluster_id")["tid"]
            #     .nunique()
            # )
            #
            # for cluster_id, count in non_none_counts.items():
            #     print(f"\tcluster {cluster_id} - {count} active tid")





if __name__ == '__main__':
    main()