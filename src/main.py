import features as f
# Features extraction for each subject csv here. This is what we want to time
# Combination into the test and train data sets using testtrain.json

EXTRACT_FEATURES = True
PATH_TO_CSVS = 'csvs/raw'

def main(ft: bool, pt: str) -> None:
    '''
    Performs feature extraction, if called. Requires a path to the raw csvs for this to happen. Edit: PATH_TO_CSVS
    Performs machine learning, if called

    Parameters:
        ft (bool) - Bool of whether to extract features or not
        pt (str) - path to the csv files containing EEG data to extract features from
    '''
    if ft:
        time = f.feature(pt)
        print(f'{time} s to complete feature extraction')

if __name__ == "__main__":
    main(EXTRACT_FEATURES,
         PATH_TO_CSVS)