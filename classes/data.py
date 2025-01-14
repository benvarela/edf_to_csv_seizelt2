import os
import pyedflib
import resampy
import warnings


class Data:
    def __init__(
        self,
        data,
        channels: tuple[str],
        fs: tuple[int],
    ):
        """Initiate an EEG instance

        Args:
            data (List(NDArray[Shape['*, *'], float])): a list of data arrays. Each channel's data is stored as an entry in the list as a data array. For each entry, the rows are channels, columns are samples in time.
            channels (tuple[str]): tuple of channels as strings.
            fs (tuple[int]): Sampling frequency of each channel.
        """
        self.data = data
        self.channels = channels
        self.fs = fs

    @classmethod
    def loadData(
        cls,
        data_path: str,
        recording: tuple[str],
        modalities: tuple[str],
    ):
        """Instantiate a data object from an EDF file.

        Args:
            data_path (str): path to EDF file.
            recording (tuple[str]): list of recording names, in which the first element is the subject name (e.g. sub-001) and the second the recording name (e.g. run-01)
            modalities (tuple[str]): list of modalities to include in the data object. Options are 'eeg', 'ecg', 'emg' and 'mov'.
            
        Returns:
            Data: returns a Data instance containing the data of the EDF file.
        """

        data = list()
        channels = list()
        samplingFrequencies = list()

        for mod in modalities:
            if os.path.exists(os.path.join(data_path, recording[0], 'ses-01', mod)):
                edfFile = os.path.join(data_path, recording[0], 'ses-01', mod, '_'.join([recording[0], 'ses-01', 'task-szMonitoring', recording[1], mod + '.edf']))
                
                with pyedflib.EdfReader(edfFile) as edf:
                    samplingFrequencies.extend(edf.getSampleFrequencies())
                    channels.extend(edf.getSignalLabels())
                    n = edf.signals_in_file
                    for i in range(n):
                        data.append(edf.readSignal(i))
                    edf._close()
            else:
                warnings.warn('Recording ' + recording[0] + '_' + recording[1] + ' does not contain ' + mod + ' data!')

        return cls(
            data,
            channels,
            samplingFrequencies,
        )
