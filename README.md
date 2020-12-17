# Cough Classifier - CS 433 Machine Learning Project 2

This repository contains the code base for our (Nina Mainusch, Xavier Oliva, Devrim Celik) machine learning project 2
submission. In this project, we were given information about coughs in their segmented form, and it was our goal to
decide whether each sample was *dry* or *wet* cough, using supervised machine learning with expert supplied labels.

### Approaches

For this project, we developed three approaches:

* *Classical Approach*: Using common machine learning algorithms that are contained within the `sklearn` Python package.
* *Artificial Neural Network (ANN) Approach *: Using a simple feed forward neural network by the mean of the `PyTorch 1.7` Python package.
* *Convolutional + Recurrent Neural Network (CRNN) Approach*: Using a neural network on raw sound files by the mean of
  the `PyTorch 1.7` Python package. Warning: this model was not trained, can be considered work in progress.

### Structure of the Repository

* `crnn_audio/`: contains the PyTorch convolutional + recurrent neural network (CRNN) class
    * `crnn_audio/evaL/`: classes defining evaluation and inference procedure
    * `crnn_audio/net/`: contains the PyTorch CRNN class
    * `crnn_audio/train/`: classes defining training procedure
    * `crnn_audio/utils/`: contains util functions used in other folders
    * `crnn_audio/config.json`: define model architecture hyperparameters
    * `crnn_audio/config.json`: define additional hyperparameters and audio transformations used in training
    * `crnn_audio/run_lstm_raw_audio.py`: script used for testing the CRNN neural network approach

* `data/`: contains the datasets for this project, that are divided into `no`, `coarse` or `fine` segmentation
* `models/`:
    * `models/grid_search_results/`: used for storing the results of the neural network grid-search
    * `models/logs/`: used for logging
    * `models/weights`: used for saving weights
* `resources/`: contains resources that were either directly or indirectly used
* `src/`:
    * `src/model/model.py`: contains the PyTorch neural network class
    * `src/notebooks/`: contains Jupyter notebooks used for either the grid-search of the classical approach or
      exploratory data analysis
    * `src/utils/`: contains the main code base
        * `src/config.py`: global parameters
        * `src/feature_engineering.py`: feature engineering related functions
        * `src/generic.py`: TODO
        * `src/get_data.py`: data import related functions
        * `src/model_helpers.py`: functions that were used for training and evaluating models
        * `src/preprocessing.py`: preprocessing related functions
        * `src/train.py`: functions for performing hyperparameter search for the classical models
        * `src/utils.py`: generic helper functions
* `vis/`: visualizations
* `run_classic.py`: script used for testing the artificial neural network approach (TODO is that right?)
* `run_deep.py`: script used for training/testing the artificial neural network approach
* `run_deep_all.sh`: shell script used in order to perform a grid search on the neural network approach

### Usage

Given that all the requirements are installed (for which we of course recommend a virtual environment), e.g., using

```
pip3 install -r requirements.txt
```

this is how both approaches can be executed.

###### Classical Approach
For executing a complete grid-search, the user has to run the notebooks found in `src/notebooks/grid_search_classical`.
There are 6 files in total, with the different cases:

For every segmentation type (`no`, `coarse`, `fine`):
* whether the extra user-metadata is supposed to be used

After running the notebook, the best results are given in `.pkl` files that can be visualized in `src/grid_search_classical_Classical_Model_Results.ipynb`.

To create the final submissions, the user updates the best models from the results and updates the `run_classical.py` with the according models and hyperparameters and runs the file in the root folder of the project:

```
$ python3 run_classical.py
```

The test results are automatically saved in the folder `data/test/predictions_classical/`

###### ANN Approach

For executing a complete grid-search, in the sense that one might also want to iterate over...

* whether to train one model for each expert or not
* whether the extra user-metadata is supposed to be used

the shell script should be used. In order to properly execute it, one has to first make it executable, i.e.

```
$ chmod +x run_deep_all.sh
```

at which point it can be executed via

```
$ ./run_deep_all.sh
```

Otherwise, one can directly execute the corresponding script using

```
$ python3 run_deep.py arg1 arg2 arg3
```

where

* `arg1` (boolean): if `True` one model is trained for each expert. Otherwise, if `False`, one model is trained for the
  whole dataset.
* `arg2` (boolean): if `True` the user metadata is dropped and not used. Otherwise, if `False`, the model makes use of
  them.
* `arg3` (boolean): if `True` grid search is performed. Otherwise, if `False`, a typical *training/testing* setup is
  executed.
  
###### CRNN Approach

First, one has to prepare the raw audio sounds. To do that, the user can download the files on the [`COUGHVID crowdsourcing dataset`](https://zenodo.org/record/4048312#.X4laBNAzY2w) to the `crnn_audio/data/` folder.

As the CRNN network only accepts certain audio file formats, the user has to run:

* ` python3 crnn_audio/data/convert_to_wav.py`
  
to convert the raw audio sunds to the accepted `.wav` format. Only the files available in the annotated dataset will be converted and saved to `crnn_audio/data/wav_data/`.
(`ffmpeg` has to be installed to successfully run this file)

Secondly, the absolute path to the data folder has to be updated in `config.json`:

By default,
`"path": "/home/xavi_oliva/Documents/EPFL/Projects/cs-433-project-2-cough_classifier/crnn_audio/data/"`

For further details on how to train and test the model, refer to the dedicated file:

[crnn_audio/README.md](https://github.com/CS-433/cs-433-project-2-cough_classifier/blob/master/crnn_audio/README.md)
