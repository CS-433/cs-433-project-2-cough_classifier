# Cough Classifier - CS 433 Machine Learning Project 2

This repository contains the code base for our (Nina Mainusch, Xavier Oliva, Devrim Celik) machine learning project 2 submission. In this project, we were given information about coughs in their segmented form, and it was our goal to decide whether each sample was *dry* or *wet* cough, using supervised machine learning with expert supplied labels.

### Approaches
For this project, we developed two approaches:
* *Classical Approach*: Using common machine learning algorithms that are contained within the `sklearn` Python package.
* *Neural Network Approach*: Using a simple feed forward neural network by the mean of the `PyTorch 1.7` Python package.

### Structure of the Repository
* `crnn_audio/`: TODO
* `data/`: contains the datasets for this project, that are divided into `no`, `coarse` or `fine` segmentation
* `models/`:
  * `models/grid_search_results/`: used for storing the results of the neural network grid-search
  * `models/logs/`: used for logging
  * `models/weights`: used for saving weights
* `resources/`: contains resources that were either directly or indirectly used
* `src/`:
  * `src/model/model.py`: contains the PyTorch neural network class
  * `src/notebooks/`: contains Jupyter notebooks used for either the grid-search of the classical approach or exploratory data analysis
  * `src/utils/`: contains the main code base
    * `src/config.py`: global parameters
    * `src/feature_engineering.py`: feature engineering related functions
    * `src/generic.py`: TODO
    * `src/get_data.py`: data import related functions
    * `src/model_helpers.py`: functions that were used for training and evaluating models
    * `src/preprocessing.py`: preprocessing related functions
    * `src/train.py`: functions for performing hyperparameter search for the classical models
    * `src/utils.py`: generic helper functions
* `vis/`: vizualizations
* `run_classic.py`: script used for testing the artificial neural network approach (TODO is that right?)
* `run_deep.py`: script used for training/testing the artificial neural network approach
* `run_deep_all.sh`: shell script used in order to perform a gridsearch on the neural network approach
* `setup.py`: TODO


### Usage
Given that all the requirements are installed (for which we of course recommend a virtual environment), e.g., using
```
pip3 install -r requirements.txt
```
this is how both approaches can be executed.

###### Classical Approach
TODO

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
* `arg1` (boolean): if `True` one model is trained for each expert. Otherwise, if `False`, one model is trained for the whole dataset.
* `arg2` (boolean): if `True` the user metadata is dropped and not used. Otherwise, if `False`, the model makes use of them.
* `arg3` (boolean): if `True` gridsearch is performed. Otherwise, if `False`, a typical *training/testing* setup is executed.
