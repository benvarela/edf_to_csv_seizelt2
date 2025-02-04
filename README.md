# README

This repository contains an example of the code to load the [SeizeIT2 dataset](https://openneuro.org/datasets/ds005873) and to train the model included in the [dataset paper](https://arxiv.org/abs/2502.01224).

# loader_test.py
Script with an example for loading files from the dataset. The classes classes.data and classes.annotation are used to create a data object, containing the signal data and extra information,  and an annotation object, containing all information regarding the seizure events of the recording.

# main_net.py
Script to train and evaluate the ChronoNet model with all parameters as in the paper. This is a suggestion of a framework that uses the data loaders and a Keras implementation of the training and evaluation routines. The data generators are likely to take a long time to run (arround 3 hours), hence the option to save the training and validation generators and load them in future runs.

## Conda environment setup
The python packages (and corresponding versions) used in the development of the scripts in this repository are gathered in 'environment.yml'. To easily create a conda environment with the same package versions to run the code, follow the instructions below:
```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda env create -n ENV_NAME -f environment.yml
```
