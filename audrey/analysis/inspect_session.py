import sys
sys.path.append('.')
from audrey.analysis.analysis_params import dataset_dir
from audrey.analysis.data_io import DataIO

ID_TO_INSPECT = '251014_B'


def main():
    data_io = DataIO(dataset_dir)

    print('Sessions found in dataset:')
    if len(data_io.sessions):
        for s in data_io.sessions:
            print(f'\t- {s}')
    else:
        print(f'\tNO SESSIONS FOUND')
        return
    
    data_io.load_session(ID_TO_INSPECT, load_pickle=True)
    data_io.dump_as_pickle()

    print(f'\n\nSession contains recordings:')
    for r in data_io.recording_ids:
        bursts_rec = data_io.burst_df.loc[data_io.burst_df['rec_id'] == r]
        n_trains = bursts_rec['train_id'].nunique()
        n_cells = len(data_io.spiketimes[r])
        stimtype = bursts_rec.stimtype.unique()
        assert len(stimtype) == 1
        stimtype = stimtype[0]

        print(f'\t{r}')
        print(f'\t\tn bursts: {bursts_rec.shape[0]}')
        print(f'\t\tn trains: {n_trains}')
        print(f'\t\tn cells: {n_cells}')
        print(f'\t\tstimtype: {stimtype}')

        print(f'\n')
    

if __name__ == '__main__':
    main()

    

