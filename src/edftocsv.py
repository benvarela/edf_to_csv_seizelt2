'''
This file takes the raw SeizelT2 dataset and outputs the channel that were interested in in compressed csv files
'''

from importlib import resources as impresources
from pathlib import Path
from classes.data import Data
from classes.annotation import Annotation
import numpy as np
import gzip

# Insert your path to the dataset here
path_to_dataset = '../Dataset/'
SUBJECTS = 129
EEG_CHANNEL_1 = 'BTEleft SD'
EEG_CHANNEL_2 = 'BTEright SD'
HZ = 256

# Initialisation of function that extracts data from a single run, once the channel is confirmed and the data is extracted from the .edf
def pull_channel_run(EEG_CHANNEL: str, d: Data, a: Annotation, sub_dict: dict, rec) -> dict:
    # Format into a numpy array
    i = d.channels.index(EEG_CHANNEL)
    run_data = d.data[i]
    run_time = np.arange(run_data.shape[0]) * (1/HZ)
    run_sz = np.zeros(run_data.shape[0])
    for times in a.events:
        onset = times[0]
        offset = times[1]
        istart = np.where(run_time == onset)[0][0]
        iend = np.where(run_time == offset)[0][0]
        run_sz[istart:iend] = 1

    # Add to the sub_dict
    sub_dict[rec[1]] = np.stack([run_data, run_sz])
    sub_dict['run_list'].append(rec[1])
    return sub_dict

## Build recordings list - variable recordings has the structure ['sub-001', 'run-01'], for all subjects and runs
data_path = Path(path_to_dataset)
sub_list = [x for x in data_path.glob("sub*")]
recordings = [[x.name, xx.name.split('_')[-2]] for x in sub_list for xx in (x / 'ses-01' / 'eeg').glob("*edf")]

# Loop over all of the subjects
for sub in np.array(range(SUBJECTS)) + 1:
    sub_str = f'sub-{sub:03d}'
    sub_dict = {'subject': sub_str, 'run_list': []}

    # Loop over the run files associated with the subject
    for rec in recordings:
        if rec[0] == sub_str:
            # Access the data from that run
            d = Data.loadData(data_path.as_posix(), rec, modalities=['eeg'])
            a = Annotation.loadAnnotation(data_path.as_posix(), rec)

            # Using the desired EEG channel
            if EEG_CHANNEL_1 in d.channels:
                sub_dict = pull_channel_run(EEG_CHANNEL_1, d, a, sub_dict, rec)
            elif EEG_CHANNEL_2 in d.channels:
                sub_dict = pull_channel_run(EEG_CHANNEL_2, d, a, sub_dict, rec)

    # Check whether any data was added, before attempting to export data. No data is added in the event none of the runs associated with a patient have desired EEG channel specified in EEG_CHANNEL
    if sub_dict['run_list']:
        # Concatenate runs if two or more were extracted:
        if len(sub_dict['run_list']) >= 2:
            # Concatenate all of the runs together into a numpy array
            all_data = np.concatenate([sub_dict[run] for run in sub_dict['run_list']], axis=1)

            # Over-write the time row
            all_data[1, :] = np.arange(all_data.shape[1]) * (1/HZ)

        # Otherwise, just extract the data if only one run was extracted
        else:
            all_data = sub_dict[sub_dict['run_list'][0]]

        # Transpose, reduce to float32, and export as compressed .csv.gz
        with gzip.open(f'csvs/{sub_str}.csv.gz', 'wt') as f:
            np.savetxt(f, all_data.T.astype(np.float32), delimiter=',', header='EEG,seizure', comments='')

## Useful lines to copy and paste to interface with the classes
#rec_data = Data.loadData(data_path.as_posix(), rec, modalities=['eeg'])
#rec_annotations = Annotation.loadAnnotation(data_path.as_posix(), rec)