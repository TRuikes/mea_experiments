To run the pipeeline data should be stored as following, where `D` is the data directory. 

Recording names should contain the following:
* recording number indicating between the 2nd pair of `_ _`.

 **This nr is unique, and cannot be shared between files**
* indicate stimulation type using:
    * `pa` or `PA` anywhere in the title, if using PA
    * `DMD` or `light` anywhere in the title, if using DMD

Example format:
`YYMMDD_SNR_RNR_STIM_STIM2_anything_else_you_want_to_add`
Where:
* `YYMMDD` = Year Month Date
* `SNR` = Sample Number ('A', 'B', 'topleft', 'topright')
* `RNR` = Recording number, e.g. 003
* `STIM` = Stimulus type, choose from: (`PA`, `pa`, `DMD`, `dmd`, `PADMD`). These do not have to be positional, but need to be these exact letters.

Example recording name `251014_A_003_PA_DMD_noblocker_awesomerecording`

Misc notes:
* Do not edit names of `.raw` files after sorting. The pipelines uses the recording names in the sorting file to detect `.raw` files
* Multiple recordings on a single day are stored in unique
data folders
* Any data ommitted in the list below is not necessary for the pipeline
* Only 1 subdirectory should be in ~/sorted: the one containing the .GUI directory
* Make an empty 'processed' dir

```
D/raw
D/csv
    SID_DMD_position.csv
    SID_MEA_position.csv
    RX_trials.csv
    RX_trials.csv
    ...
/sorted  
    F1.raw
    F2.raw
    FN.raw

    /spikesortingexport
        /SOMENAME.GUI
            phy_gui_file_1
            phy_gui_file_2
            etc.  
    
D/processed

```