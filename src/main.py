import features as f
import concatenate as c
import traintest as t

## Settings
EXTRACT_FEATURES = False
CONCATENATE_FEATURES = False
RUN_ALGORITHMS = True
PATH_TO_RAW_CSVS = 'csvs/raw'
PATH_TO_FEATURE_CSVS = 'csvs/features'
PATH_TO_CONCATENATED_CSVS = 'csvs/traintest'

## Computer cores you want this to use
CORES = 6

## Main function
def main(ft: bool, ft_cc: bool, tt: bool, pt_raw: str = None, pt_ft: str = None, pt_tt: str = None, cores: int = 1) -> None:
    '''
    Performs feature extraction, if called. Requires a path to the raw csvs for this to happen. Edit: PATH_TO_RAW_CSVS
    Performs concatenation of the feature csvs, if called. Requires a path to the feature csvs for this to happen. Edit: PATH_TO_FEATURE_CSVS
    Performs machine learning, if called. Requires a path to the concatenated features csvs for this to work

    Parameters:
        ft (bool) - Bool of whether to extract features or not
        ft_cc (bool) - Bool of whether to concatenate the feature csvs together
        tt (bool) - Bool of whether to run the machine learning algorithms
        pt_raw (str) - path to the csv files containing EEG data to extract features from
        pt_ft (str) - path to where the csvs files containing the feature data is
        pt_tt(str) - path to where the concatenated train and test feature sets are
    '''
    if ft:
        time = f.feature(pt_raw)
        print(f'{time} s to complete feature extraction')
    if ft_cc:
        c.concatenate(pt_ft)
    if tt:
        t.traintest(pt_tt, cores)

if __name__ == "__main__":
    main(EXTRACT_FEATURES,
         CONCATENATE_FEATURES,
         RUN_ALGORITHMS,
         PATH_TO_RAW_CSVS,
         PATH_TO_FEATURE_CSVS,
         PATH_TO_CONCATENATED_CSVS,
         CORES)