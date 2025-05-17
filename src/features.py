'''
This file facillitates the data smoothing, segmentation, and finally feature extraction required to output a csv of features for all of the subject's eeg data.
'''

from time import process_time
from pathlib import Path
import pandas as pd
import numpy as np
import json
from scipy.signal import savgol_filter
from pywt import WaveletPacket
from scipy.stats import skew, kurtosis
import gzip


def process_run_indexes(run_ends: list) -> list:
    '''
    Breifly preprocess run indexes, so the start and end index of runs is output in a list, as oppoosed to just the end indexes
    Note that these indexes are prepared to work immediately with slice notation; +1/-1 business has already been thought of and accounted for

    INPUT: run_ends: list of all the indexes for when a run ends
    OUTPUT: out: list containing lists of [i_start, i_end] for each run
    '''
    # Initialise output list, length and negative indexes to iterate over
    out = list()
    for i in range(len(run_ends)):
        # Add the start and end indexes sequentially to the new output list
        if i == 0:
            out.append([0, run_ends[i]])
        else:
            out.append([run_ends[i-1], run_ends[i]])
    return out


def smooth_run(run_index: list, eeg: np.ndarray, sz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    This function applies a Savitzky-Golay filter with N = 22, M = 35 to a run sliced from the concatenated eeg runs for a subject

    INPUTS - run_index: list contains the start and end indexes of the run, eeg: np.ndarray contains the full eeg data run for a patiet, sz: np.ndarray contains the seizure labels for each time point
    OUTPUTS - run_smooth: np.ndarray the smoothed slice cut out from full patient eeg data
    '''
    # Cut the run out, and the labels out
    run_slice = eeg[run_index[0]:run_index[1]]
    sz_slice = sz[run_index[0]:run_index[1]]

    # Smooth the slice with the SG (22, 35) filter
    run_smooth = savgol_filter(run_slice, window_length=35, polyorder=22)
    
    return run_smooth, sz_slice

def segment_run(run_smooth: np.ndarray, sz_slice: np.ndarray) -> list:
    '''
    Segments the run into 1 second, 50% overlap chunks. Exclusion of segments that overlap a seizure and a non-seizure region.

    INPUT: run_smooth: np.ndarray the smoothed run to be segmented, sz: np.ndarray the seizure labels associated with each eeg value
    OUTPUT: run_segmented - list in format [[label: 0 or 1, segment: np.ndarray], other segments...]
    '''
    length = run_smooth.shape[0]
    start = 0
    end = 256   
    run_segmented = list()
    
    while end <= length:
        segment = run_smooth[start:end]
        start_label, end_label = sz_slice[start], sz_slice[end-1]
        if start_label == end_label:
            run_segmented.append([start_label, segment])
        start += 128
        end += 128

    return run_segmented

def entropy(band: np.ndarray):
    '''
    Returns the entropy of an array from a frequency band that has undergone wavelet decomposition
    
    Parameters:
        band (np.ndarray): input wavelet band

    Returns:
        entropy: shannon entropy of the input wavelet band
    '''
    band_sq  = band ** 2
    energy = np.sum(band_sq)
    prob = band_sq / energy
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def run_features(run_segmented: list) -> pd.DataFrame:
    '''
    Extracts features from each segment using wavelet decomposition with weighted node combination
    to precisely isolate alpha (8-13Hz) and beta (13-32Hz) bands. Returns a DataFrame with
    12 features per segment (mean, stdev, skew, kurtosis, energy, entropy for each band).

    INPUTS: 
        run_segmented: list of segments in format [[label: 0 or 1, segment: np.ndarray], ...]
    
    OUTPUTS: 
        pd.DataFrame with features and labels (one row per segment)
    '''
    # Initialize feature storage
    features = []
    
    for segment in run_segmented:
        label, smooth_run = segment[0], segment[1]

        try:
            # Perform wavelet decomposition
            wp = WaveletPacket(data=smooth_run, wavelet='db8', mode='symmetric', maxlevel=5)
            
            # Reconstruct bands with weighted node combination
            # Alpha (8-13Hz) = aaada (8-12Hz) + 25% of aaadd (12-16Hz)
            alpha_full = wp['aaada'].data
            alpha_partial = 0.25 * wp['aaadd'].data
            alpha_signal = alpha_full + alpha_partial
            
            # Beta (13-32Hz) = 75% of aaadd (13-16Hz) + aadaa (16-20Hz) + aadad (20-24Hz) + aadda (24-28Hz)
            beta_partial = 0.75 * wp['aaadd'].data
            beta_signal = (beta_partial + wp['aadaa'].data + wp['aadad'].data + wp['aadda'].data)

            # Extract features 
            feature_row = [
                # Alpha features
                np.mean(alpha_signal), np.std(alpha_signal), 
                skew(alpha_signal), kurtosis(alpha_signal),
                np.sum(alpha_signal ** 2), entropy(alpha_signal),
                
                # Beta features
                np.mean(beta_signal), np.std(beta_signal),
                skew(beta_signal), kurtosis(beta_signal),
                np.sum(beta_signal ** 2), entropy(beta_signal),
                
                # Seizure label
                label
            ]

            features.append(feature_row)
            
        except Exception as e:
            print(f"Skipping segment (length={len(smooth_run)}): {e}")
            continue

    # Convert to DataFrame
    columns = [
        'alpha_mean', 'alpha_stdev', 'alpha_skew', 'alpha_kurtosis', 
        'alpha_energy', 'alpha_entropy',
        'beta_mean', 'beta_stdev', 'beta_skew', 'beta_kurtosis', 
        'beta_energy', 'beta_entropy',
        'label'
    ]
    return pd.DataFrame(features, columns=columns)

def export_csv(df_features: pd.DataFrame, sub: str):
    df_features.to_csv(f'csvs/features/{sub}.csv.gz', index=False, compression='gzip')

def feature(path_to_csvs: str):
    ## Initialisations
    # Initialise feature extraction run-time
    time = 0
    # Initialise csv paths
    csvs = [f'{path_to_csvs}/{x}' for x in Path(path_to_csvs).glob("sub*")]
    # Initialise segmentation indexes, in a dictionary
    with open('json/segments.json', 'r') as j:
        segment_indexes = json.load(j)

    # Loop over all subject csvs to extract features from stored EEG data
    for csv in csvs:
        # Read data in, numpy arrays
        df = pd.read_csv(csv, compression = 'gzip')
        eeg, sz = df['EEG'].to_numpy(), df['seizure'].to_numpy()

        # Determine which subject is being analysed, and their segmentation index list
        sub = csv.split('/')[1][:7]
        isub = segment_indexes['subject'].index(sub)
        run_indexes = process_run_indexes([int(x) for x in segment_indexes['run_start_indexes'][isub]])

        # Process each run, with smoothing, segmentation and feature extraction
        for run_index in run_indexes:
            t1 = process_time()
            run_smooth, sz_slice = smooth_run(run_index, eeg, sz)
            run_segmented = segment_run(run_smooth, sz_slice)
            df_features = run_features(run_segmented)
            t2 = process_time()

            # Output as a compressed csv for storage
            export_csv(df_features, sub)
            time += t2 - t1

            # log
            print(f"Processed {sub} run {run_index} in {t2-t1:.2f}s")

    return time