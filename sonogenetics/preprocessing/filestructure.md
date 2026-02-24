remarks:
* denote blocker as: 'noblocker', 'washout', 'cppcnqx'

~SESSION_id/
    processed/
        sorted/     # output from spikesorting pipeline

        # output from this pipeline
        triggers.h5
        spiketimes.h5
        cluster_info.csv
        waveforms.h5
        figures/
        trials.csv



    raw/
        YYYY-MM-DD_MEA_position.csv
        YYYY-MM-DD_onda_laser_calibration.json
        probefile.prb

        rec_1_STIMDETAILS.raw
        rec_2_STIMDETAILS.raw
        ...
        rec_N_STIMDETAILS.raw

        rec_1_STIMDETAILS_trials.csv
        rec_2_STIMDETAILS_trials.csv
        ...
        rec_N_STIMDETAILS_trials.csv

        PROTOCOL_1.py
        PROTOCOL_2.py


