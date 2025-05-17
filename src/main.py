import features as f
# Features extraction for each subject csv here. This is what we want to time
# Combination into the test and train data sets using testtrain.json

EXTRACT_FEATURES = True
PATH_TO_CSVS = 'csvs/raw'

def main(ft: bool, pt: str) -> None:
    '''
    Performs feature extraction, if called. Requires a path to the raw csvs for this to happen. Edit: PATH_TO_CSVS
    Performs machine learning, if called
    '''
    if ft:
        time = f.feature(pt)

if __name__ == "__main__":
    main(EXTRACT_FEATURES,
         PATH_TO_CSVS)