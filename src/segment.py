'''
This file outputs the indexes where new runs start, as concatenated in the compressed csv files that are output by edftocsv
The output json file is to be used after filtering, and before feature extraction so that segmenting can occur correctly.

The json file has the same structure as the variable seg_sub: seg_sub = {'subject': list(), 'run_start_indexes': list()}
Note that the 'run_start_indexes' are stored in such a way that each index is the index immediately after a run ends, INCLUDING the final run
'''

from importlib import resources as impresources
import numpy as np
import pandas as pd
from pathlib import Path
from classes.data import Data
from classes.annotation import Annotation
import json

# Insert your path to the dataset here
path_to_dataset = '../Dataset/'
SUBJECTS = 129
EEG_CHANNEL_1 = 'BTEleft SD'
EEG_CHANNEL_2 = 'BTEright SD'

## Build recordings list - variable recordings has the structure ['sub-001', 'run-01'], for all subjects and runs
    # This helps you to interface with the way the Data and Annotation classes expect to be instantiated
data_path = Path(path_to_dataset)
sub_list = [x for x in data_path.glob("sub*")]
recordings = [[x.name, xx.name.split('_')[-2]] for x in sub_list for xx in (x / 'ses-01' / 'eeg').glob("*edf")]

seg_sub = {'subject': list(), 'run_start_indexes': list()}

for sub in np.array(range(SUBJECTS)) + 1:
    sub_str = f'sub-{sub:03d}'
    # Temporary store of the duraction of each run
    run_durations = list()

    # Loop over the run files associated with the subject
    for rec in recordings:
        if rec[0] == sub_str:
            # Get all of the run recording durations.
            a = Annotation.loadAnnotation(data_path.as_posix(), rec)
            run_durations.append(a.rec_duration)

    # Output populate 'run_start_idnexes' using run_durations
    l = len(run_durations)
    np_dur = np.zeros(l)
    for i in range(l):
        j = 0
        while j <= i:
            np_dur[i] += run_durations[j]
            j += 1

    # Add to the seg_sub dictionary
    seg_sub['subject'].append(sub_str)
    seg_sub['run_start_indexes'].append(list(np_dur * 256))

#Output as json
with open('json/segments.json', 'w') as j:
    json.dump(seg_sub, j)