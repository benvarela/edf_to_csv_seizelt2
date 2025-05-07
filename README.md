# README

This repository is a branch of the Github used to process all imaging modalities included in the [SeizeIT2 dataset](https://openneuro.org/datasets/ds005873) ([link]https://github.com/biomedepi/seizeit2).

"main.py" is a modified version of the originally named loader_test.py, designed to instead use all the .edf files to construct a single .csv file. This file uses the classes stored in 'classes', which have remained unmodified.

# CSV Output

This csv file will have the following columns the full length of recording duration:
* EEG: contains the bte-EEG readings from the cross brain channel for all participants
* time(s): the time of the recording 
* seizure: contains 0 where a seizure is not ocurring, and 1 where a seizure it occurring

# Dependencies

The dependencies have also been modified, and are available in requirements.txt