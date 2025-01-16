import numpy as np
from tensorflow import keras
from net.utils import apply_preprocess_eeg
from tqdm import tqdm
from classes.data import Data


class SequentialGenerator(keras.utils.Sequence):
    ''' Class where a keras sequential data generator is built (the data segments are continuous and aligned in time).

    Args:
        config (cls): config object with the experiment parameters
        recs (list[list[str]]): list of recordings in the format [sub-xxx, run-xx]
        segments: list of keys (each key is a list [1x4] containing the recording index in the rec list,
                  the start and stop of the segment in seconds and the label of the segment)
        batch_size: batch size of the generator
        shuffle: boolean, if True, the segments are randomly mixed in every batch
    
    '''

    def __init__(self, config, recs, segments, batch_size=32, shuffle=False, verbose=True):
        
        'Initialization'
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data_segs = np.empty(shape=[len(segments), int(config.frame*config.fs), config.CH])
        self.labels = np.empty(shape=[len(segments), 2])
        self.verbose = verbose
        
        pbar = tqdm(total = len(segments)+1, disable = not self.verbose)

        count = 0
        prev_rec = int(segments[0][0])

        rec_data = Data.loadData(config.data_path, recs[prev_rec], modalities=['eeg'])
        rec_data = apply_preprocess_eeg(config, rec_data)
        
        for s in segments:
            curr_rec = int(s[0])
            
            if curr_rec != prev_rec:
                rec_data = Data.loadData(config.data_path, recs[curr_rec], modalities=['eeg'])
                rec_data = apply_preprocess_eeg(config, rec_data)
                prev_rec = curr_rec

            start_seg = int(s[1]*config.fs)
            stop_seg = int(s[2]*config.fs)

            if stop_seg > len(rec_data[0]):
                self.data_segs[count, :, 0] = np.zeros(config.fs*config.frame)
                self.data_segs[count, :, 1] = np.zeros(config.fs*config.frame)
            else:
                self.data_segs[count, :, 0] = rec_data[0][start_seg:stop_seg]
                self.data_segs[count, :, 1] = rec_data[1][start_seg:stop_seg]

            if s[3] == 1:
                self.labels[count, :] = [0, 1]
            elif s[3] == 0:
                self.labels[count, :] = [1, 0]

            count += 1
            pbar.update(1)

        
        self.key_array = np.arange(len(self.labels))

        self.on_epoch_end()


    def __len__(self):
        return len(self.key_array) // self.batch_size

    def __getitem__(self, index):
        keys = np.arange(start=index * self.batch_size, stop=(index + 1) * self.batch_size)
        x, y = self.__data_generation__(keys)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        if self.config.model == 'DeepConvNet' or self.config.model == 'EEGnet':
            out = self.data_segs[self.key_array[keys], :, :, np.newaxis].transpose(0,2,1,3), self.labels[self.key_array[keys]]
        else:
            out = self.data_segs[self.key_array[keys], :, :], self.labels[self.key_array[keys]]
        return out



class SegmentedGenerator(keras.utils.Sequence):
    ''' Class where the keras segmented data generator is built, implemented as a more efficient way to load segments that were subsampled from multiple recordings.

    Args:
        config (cls): config object with the experiment parameters
        recs (list[list[str]]): list of recordings in the format [sub-xxx, run-xx]
        segments: list of keys (each key is a list [1x4] containing the recording index in the rec list,
                  the start and stop of the segment in seconds and the label of the segment)
        batch_size: batch size of the generator
        shuffle: boolean, if True, the segments are randomly mixed in every batch
    
    '''

    def __init__(self, config, recs, segments, batch_size=32, shuffle=True, verbose=True):
        
        'Initialization'
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose

        self.data_segs = np.empty(shape=[len(segments), int(config.frame*config.fs), config.CH])
        self.labels = np.empty(shape=[len(segments), 2])
        segs_to_load = segments

        pbar = tqdm(total = len(segs_to_load)+1, disable=self.verbose)
        count = 0

        while segs_to_load:

            curr_rec = int(segs_to_load[0][0])
            comm_recs = [i for i, x in enumerate(segs_to_load) if x[0] == curr_rec]

            rec_data = Data.loadData(config.data_path, recs[curr_rec], modalities=['eeg'])
            rec_data = apply_preprocess_eeg(config, rec_data)

            for r in comm_recs:
                start_seg = int(segs_to_load[r][1]*config.fs)
                stop_seg = int(segs_to_load[r][2]*config.fs)

                self.data_segs[count, :, 0] = rec_data[0][start_seg:stop_seg]
                self.data_segs[count, :, 1] = rec_data[1][start_seg:stop_seg]

                if segs_to_load[r][3] == 1:
                    self.labels[count, :] = [0, 1]
                elif segs_to_load[r][3] == 0:
                    self.labels[count, :] = [1, 0]
                
                count += 1
                pbar.update(1)
                
            segs_to_load = [s for i, s in enumerate(segs_to_load) if i not in comm_recs]
        
        self.key_array = np.arange(len(self.labels))

        self.on_epoch_end()


    def __len__(self):
        return len(self.key_array) // self.batch_size

    def __getitem__(self, index):
        keys = np.arange(start=index * self.batch_size, stop=(index + 1) * self.batch_size)
        x, y = self.__data_generation__(keys)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        if self.config.model == 'DeepConvNet' or self.config.model == 'EEGnet':
            out = self.data_segs[self.key_array[keys], :, :, np.newaxis].transpose(0,2,1,3), self.labels[self.key_array[keys]]
        else:
            out = self.data_segs[self.key_array[keys], :, :], self.labels[self.key_array[keys]]
        return out


    
    
