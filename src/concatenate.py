'''
This file takes the csvs that are output from features.py, which are individually output for each subject,
and concatenates them such that a single csv containing all of the training data is formed, and a single csv
containing all of the testing data is formed.

This file now also ensures that a 90:10 non-seizure to seizure data balance is achieved, by only adding
non-seizure data around the seizure data in such a way that this proportion is met. All other regions
are left out.

The subjects to be included in the training and testing concatenated data sets is included in the manually written
testtrain.json. 

INPUTS: The feature csvs output for each subject. Stored csvs/features, sub-XXX.csv.gz
OUTPUTS: two csvs, stored csvs/testtrain, test.csv.gz and train.csv.gz
'''

import json
import pandas
from pathlib import Path
import pandas as pd
import numpy as np

def balance(csv_path) -> pd.DataFrame:
    '''
    Outputs a dataframe that only contains seizure data in a 90:10 non-seizure to seizure ratio, by splicing
    out regions that are a long time between seziures as a priority

    Parameters:
        - csv_path (str): path to the feature csv to balance/splice data out of

    Returns:
        - data (pd.DataFrame): the altered dataframe with a much better balance, to be concatenated later all together
    '''
    ## Read the data in
    df = pd.read_csv(csv_path, compression = 'gzip')

    ## Row indexes where the label == 1
    sz_indexes = np.argwhere(df['label'] == 1).flatten()

    ## Separate the row indexes into sublists corresponding to independent seizures
    sz_fullslices = list()
    start = 0
    for i in range(1, len(sz_indexes)):
        # If there's a jump, then there's a slice. Extract and update the start
        if sz_indexes[i] != sz_indexes[i-1] + 1:
            sz_fullslices.append(sz_indexes[start:i])
            start = i
    # Final slice included
    sz_fullslices.append(sz_indexes[start:])

    ## Create a set of the final indexes to extract
    balanced_indexes = set()
    for slice in sz_fullslices:
        # Find size of slice
        l = len(slice)

        # Create new bounds to include 9x more data, 4.5x either side of the seizure
        lower = slice[0] - int(4.5*l)
        upper = slice[-1] + int(4.5*l)

        # Edit if the bounds exceed the end or are before the start
        if lower < 0:
            lower = 0
        if upper >= len(df):
            upper = len(df) - 1

        # Construct and add the slice
        expanded_slice = np.arange(lower, upper + 1)
        balanced_indexes.update(expanded_slice)

    ## Convert back to a sorted lise
    balanced_indexes = sorted(list(balanced_indexes))

    ## Index out of the dataframe; .iloc[] makes this really easy!
    df_balanced = df.iloc[balanced_indexes].reset_index(drop=True)

    ## Log
    print(f'Finished {csv_path.name}')

    return df_balanced

def concatenate(path_to_csvs: str) -> None:
    '''
    Performs concatenation as described in the file description at the top of the file
    '''
    ## Import the json data
    with open('json/testtrain.json', 'r') as j:
            tt_dict = json.load(j)

    ## Initialise the lists for which subject to use for test or train, initialise the lists containing the data to concatenate
    test_sub = [sub for sub in tt_dict["test"] if sub not in tt_dict["test_excluded"]]
    train_sub = [sub for sub in tt_dict["train"] if sub not in tt_dict["train_excluded"]]
    test_featuredata = list()
    train_featuredata = list()

    ## Loop over all of the files, output the csvs to a list as dataframes if they're in the test or train set
    csv_paths = [x for x in Path(path_to_csvs).glob("sub*")]
    for csv_path in csv_paths:
        sub_str = csv_path.name[:7]
        if sub_str in test_sub:
            test_featuredata.append(balance(csv_path))
        elif sub_str in train_sub:
            train_featuredata.append(balance(csv_path))

    ## Concatenate all of the contained csvs together, remove extremely noisy rows, output
    test_file = pd.concat(test_featuredata, ignore_index=True)
    train_file = pd.concat(train_featuredata, ignore_index=True)
    test_file = test_file[test_file['alpha_energy'] < 1e-44].reset_index(drop=True)
    train_file = train_file[train_file['alpha_energy'] < 1e-44].reset_index(drop=True)
    test_file.to_csv('csvs/traintest/test.csv.gz', compression='gzip', index=False)
    train_file.to_csv('csvs/traintest/train.csv.gz', compression='gzip', index=False)

    return None