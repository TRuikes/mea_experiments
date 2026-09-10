import os
import pandas as pd
from sonogenetics.analysis.lib.analysis_params import dataset_dir, figure_dir_analysis
from sonogenetics.analysis.lib.data_io import DataIO
from sonogenetics.analysis.lib.analysis_tools import detect_preferred_electrode, get_params_protocol, params_abbreviation
from sonogenetics.analysis.data_list import data_list

def main():

    for session_id in data_list:
        print(f'\n\nSession: {session_id}')
        fig_save_dir = figure_dir_analysis / session_id
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)

        # Load dataset + dump as pickle to speedup future data loading
        data_io = DataIO(dataset_dir)
        data_io.load_session(session_id, load_pickle=False, load_waveforms=False)
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

        results = pd.DataFrame()
        for cid in data_io.cluster_ids:
            results.at[cid, 'response_type'] = 'none'

        for (rec_id, protocol), rdf in data_io.train_df.groupby(['rec_id', 'protocol']):
            print(f'\t-{rec_id}')


            ptcs = list(pref_ec[rec_id].keys())
            assert len(ptcs) == 1

            for cid, cinfo in pref_ec[rec_id][ptcs[0]].iterrows():
                if pd.isna(cinfo.ec):
                    continue

                else:
                    cell_rtype = results.at[cid, 'response_type']

                    ec_trialdf = rdf.query(f'electrode == {cinfo.ec}')

                    for i, r in ec_trialdf.iterrows():

                        if not cells_df.at[cid, (i, 'is_excited')] and not cells_df.at[cid, (i, 'is_inhibited')]:
                            continue

                        if r.has_dmd and r.has_laser:
                            if cell_rtype == 'none':
                                cell_rtype = 'dual'
                            elif 'dual' not in cell_rtype:
                                cell_rtype += '_dual'

                        elif not r.has_dmd and r.has_laser:
                            if cell_rtype == 'none':
                                cell_rtype = 'pa'
                            elif 'pa' not in cell_rtype:
                                cell_rtype += '_pa'



                        elif r.has_dmd and not r.has_laser:
                            if cell_rtype == 'none':
                                cell_rtype =' dmd'
                            elif 'dmd' not in cell_rtype:
                                cell_rtype += '_dmd'
                        else:
                            raise ValueError('')

                    results.at[cid, 'response_type'] = cell_rtype


        for i, r in results.iterrows():
            parts = r.response_type.split('_')
            rype = ''
            for p in sorted(parts):
                rype += f'_{p}'

            results.at[i, 'response_type'] = rype



        print_str = ''
        for k, v in results.value_counts('response_type').items():
            print_str += f'|\t{k}: {v}\t'

        print(print_str)





if __name__ == '__main__':
    main()