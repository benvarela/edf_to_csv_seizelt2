import numpy as np
from scipy.fft import fft, fftfreq
import scipy.stats as ss
import pandas as pd
import pywt as pw

def wavelet(data: np.ndarray) -> np.ndarray:
    """Computes the 1-D multilevel wavelet transform of EEG data

    Args:
        data (np.ndarray): EEG channel data over 1 second

    Returns:
        np.ndarray: [energyA, energyD5, energyD4, energyD3, energyD2,
                    entropyA, entropyD5, entropyD4, entropyD3, entropyD2]
    
        with A : 0 - 4 Hz (Delta),
             D5 : 4 - 8 Hz (Theta), 
             D4 : 8 - 16 Hz (Alpha), 
             D3 : 16 - 32 Hz (Beta), 
             D2 : 32 - 64 Hz (Gamma)
    """    
    ### Computes wavelet transform
    wavelet = 'db4'
    levels = 5
    coeffs = pw.wavedec(data, wavelet, level=levels)

    ## Initialise array for results
    results = np.zeros(shape=(5,2))

    for i, coeff in enumerate(coeffs):
        if i == 5:
            continue        ## Do not compute D1 bandwidth
        else:
            coeff_square  = coeff ** 2
            energy = np.sum(coeff_square)       ## Total energy of the bandwidth
            prob = coeff_square / energy
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log2(prob))     ## Total entropy of the bandwidth
            results[i,:] = np.array([energy, entropy])

    results = np.reshape(results, (10,1), order="F")

    return results

# def power_spectral_density(data : np.ndarray) -> float:

#     n = len(data)
#     s = 1 / 256
#     yf = fft(data)
#     xf = fftfreq(n, s)[:n//2]

#     PSD = s / n * np.abs(yf[:n//2])

#     # xf = xf[xf > 1]
#     # PSD = PSD[len(PSD) - len(xf):]

#     plt.plot(xf, PSD)
#     plt.show()
#     return


def get_features(data) -> np.ndarray:
    """Returns the features : - of the sample 
    """ 
    f0 = np.mean(data)
    f1 = np.var(data)
    f2 = ss.kurtosis(data)
    f3 = ss.skew(data)
    f_4_to_14 = wavelet(data)

    features = np.concatenate((np.array([f0, f1, f2, f3])[:,np.newaxis], f_4_to_14), axis=0).reshape((14,))

    return features

if __name__ == "__main__":
    # features = get_features(eeg[0:256])

    # print(features)
    pass








