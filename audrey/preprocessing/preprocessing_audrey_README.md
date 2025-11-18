To run the pipeeline data should be stored as following, where `D` is the data directory. 

Recording names should contain the following:
* recording number indicating between the 1st pair of `_RN_`. **This nr is unique, and cannot be shared between files**
* indicate stimulation type using:
    * `pa` or `PA` anywhere in the title, if using PA
    * `DMD` or `light` anywhere in the title, if using DMD


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