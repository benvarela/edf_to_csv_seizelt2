import pandas as pd
import numpy as np
from src.get_features import get_features

def subjects_feature(path: str) -> pd.DataFrame:
    """returns features over the entire sample space

    Args:
        path (str): path to subject EEG 

    Returns:
        pd.DataFrame:  dataframe inlcuding whether a seizure occurred and features at that time
    """
    df = pd.read_csv(path, compression = 'gzip')
    eeg =  df["EEG"].to_numpy()
    seizure = df["seizure"].to_numpy()
    threshold = 25                      ## amount of samples in each second that have to be true for seizure to be true

    N = len(eeg)
    rows = N // 256
    if N % 256 > 0:
        eeg = eeg[0 : N - N % 256]
    eeg = np.reshape(eeg, (rows, 256))

    variable = np.zeros(shape=(rows,1))
    features = np.zeros(shape=(rows, 14))

    cols = ['Seizure', 'Mean', 'Variance', 'Kurtosis', 'Skew', 'EnergyA', 'EnergyD5', 'EnergyD4', 'EnergyD3',
                           'EnergyD2', 'EntopryA', 'EntropyD5', 'EntropyD4', 'EntropyD3', 'EntropyD2']

    df_subject = pd.DataFrame(index=list(range(rows)), columns=cols)
    
    for j in range(rows):
        variable[j] = threshold > sum(seizure[ j * 256 :( j + 1 ) * 256])
        features[j,:] = get_features(eeg[j,:])

    df_subject[df_subject.columns[0]] = variable
    for k in range(1,15):
        df_subject[df_subject.columns[k]] = features[:,k - 1] 

    print(df_subject)
    return df_subject


    


df = subjects_feature('csvs/sub-001.csv.gz')