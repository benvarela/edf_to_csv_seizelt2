import os
import random
random_seed = 1
random.seed(random_seed)

import numpy as np
np.random.seed(random_seed)

import tensorflow as tf
tf.random.set_seed(random_seed)

from net import key_generator
key_generator.random.seed(random_seed)

from net import main_func

from net.DL_config import Config


###########################################
## Initialize standard config parameters ##
###########################################

## Configuration for the generator and models:
config = Config()
config.data_path = '/esat/biomeddata/SeizeIT2/bids'             # path to data
config.save_dir = 'net/save_dir'                                # save directory of intermediate and output files
if not os.path.exists(config.save_dir):
  os.mkdir(config.save_dir)
config.fs = 250                                                 # Sampling frequency of the data after post-processing
config.CH = 2                                                   # Nr of EEG channels
config.cross_validation = 'fixed'                               # validation type
config.batch_size = 128                                         # batch size
config.frame = 2                                                # window size of input segments in seconds
config.stride = 1                                               # stride between segments (of background EEG) in seconds
config.stride_s = 0.5                                           # stride between segments (of seizure EEG) in seconds
config.boundary = 0.5                                           # proportion of seizure data in a window to consider the segment in the positive class
config.factor = 5                                               # balancing factor between nr of segments in each class

## Network hyper-parameters
config.dropoutRate = 0.5
config.nb_epochs = 300
config.l2 = 0.01
config.lr = 0.01

###########################################
###########################################

##### INPUT CONFIGS:
config.model = 'ChronoNet'                                      # model architecture (you have 3: Chrononet, EEGnet, DeepConvNet)
config.dataset = 'SZ2'                                          # patients to use (check 'datasets' folder)
config.sample_type = 'subsample'                                # sampling method (subsample = remove background EEG segments)
config.add_to_name = 'test'                                     # str to add to the end of the experiment's config name

###########################################
###########################################

load_generators = False                                          # Boolean to load generators from file
save_generators = False                                         # Boolean to save the training and validation generator objects. The training generator is saved with the dataset, frame and sample type properties in the name of the file. The validation generator is always using the sequential windowed method.

main_func.train(config, load_generators, save_generators)

print('Getting predictions on the test set...')
main_func.predict(config)

print('Getting evaluation metrics...')
main_func.evaluate(config)
