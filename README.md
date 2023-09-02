# fire-forecast
Tool for forecast of fire radiative power.

## Installation
### 1. Clone the repository with:
```
git clone git@github.com:ECMWFCode4Earth/fire-forecast.git
```
### 2. Install the conda environment
To create the environment switch to the `fire-forcast` folder created by the `git clone` above and executeone of the following depending on your system:
```
# For system without GPU:
conda env create -f ci/environment_cpu.yaml
# For system with GPU
conda env create -f ci/environment_gpu.yaml
```
This will install `poetry` as well as the dependencies that are not installable with `poetry` in the new environment called `fire-forecast` (Further packages will then be managed by poetry in this environment).

### 3. Activate the environment with:
```
conda activate fire-forecast
```
### 4. Install the fire-forecast module with poetry
To install the other dependencies of the project run the following from the `fire-forcast` folder created by the `git clone`:
```
poetry install 
```
This will install all dependencies of `fire-forecast` and the module itself.
### 5. Install the pre-commit hook to automatically format and test your code before a commit:
```
pre-commit install
```

**Done!**
### Remarks
 * To test the installation you can run:
   * `pytest` in the `fire-forcast` folder created by the `git clone`
   * `pre-commit run --all-files` tests the pre-commit hook. `black` `isort` and `flake8` should be installed and the hook should be run and show: "Passed" or "Skipped" for the different modules.
 * To contribute within the project create a new branch with:
  `git checkout -b branch_name`
   and use a pull request on github.com to merge with the main branch, when you are done with the new feature.
 * To add a new dependency with `poetry`:
   * `poetry add package name` (works similar to `conda install` or `pip install`)   

## Usage
The `fire_forecast` module is subdivided into three main parts:
 1. `fire_forecast.data_retrieval`/`fire_forecast.data_preparation` for downloading/preprocessing data into various forms
 2. `fire_forecast.deep_learning` for creating, training and loading deep learning models
 3. `fire_forecast.evaluation` for evaluating models and their performance against classical methods
In the following we will give a short overview of the different parts and how to use them.

All scripts can be called with python -m fire_forecast.module_name.script_name -h to get a help message.

### 1. Data preparation
#### 1.1 Downloading data
A script to retrieve data can be found here: `fire_forecast/data_retrieval/retrieve_meteo_fire.sh`
The result is gridded data in a `netcdf` format that includes dataarrays for `frpfire` and `offire`. The models are designed, to take input in form of a fire timeseries, which are produced by the preprocessing tools.
#### 1.2 Selecting timeseries of pixels with fire
First timeseries that contain fires are extracted using the script `fire_forecast.data_preparation.SelectFirePixels`. This will produce 3 by 3 pixels wide timeseries which contain at least one recorded active fire. The usage is as follows:
```
usage: SelectFirePixels_FRP.py [-h] [--i I [I ...]] OutputPath

Script to preprocess GFAS and Meteo Data for machine learning, run script e.g. with python SelectFirePixels_FRP.py <outputpath> --i <inputdatapath1> <inputdatapath2> <inputdatapath3>

positional arguments:
  OutputPath            Enter FilePath to save output

options:
  -h, --help            show this help message and exit
  --i I [I ...], -InputPathList I [I ...]
                        Enter filepaths to inputdata (.nc fomat), needs at least a GFAS dataset with frpfire and offire variable
```
The resulting timeseries can be used for the classical approches in `fire_forecast.evaluation`.
#### 1.3 Cutting timeseries into 2 day samples for the deep learning models
The goal of our models is to predict the fire radiative power of a full day, based on the fire of the previous day, its rate of measurement and meteorological data of the full 2 days. To split the long timeseries into 2 day samples, we use the `fire_forecast.data_preparation.cut_data` script. The script takes a netcdf file with the fire timeseries and the meteorological data and splits it into 2 day samples. Additionally it splits the data into train, test and validation sets. The usage is as follows:
```
usage: cut_data.py [-h] [--output_path OUTPUT_PATH] [--fire_number_threshold FIRE_NUMBER_THRESHOLD] [--measurement_threshold MEASUREMENT_THRESHOLD] data_paths [data_paths ...]

Script to select data from dataset.

positional arguments:
  data_paths            Paths of data to read.

options:
  -h, --help            show this help message and exit
  --output_path OUTPUT_PATH
                        Path to output dataset.
  --fire_number_threshold FIRE_NUMBER_THRESHOLD
                        Fire threshold.
  --measurement_threshold MEASUREMENT_THRESHOLD
                        Measurement threshold.
```
#### 1.4 Converting the data to h5 format
The data is saved in netcdf format, which is not ideal for training deep learning models. Therefore we convert the data to h5 format with the `fire_forecast.data_preparation.create_training_set` script. The usage is as follows:
```
usage: create_training_set.py [-h] [--filename_start FILENAME_START] [--filter_nans FILTER_NANS] [--fire_number_threshold_first_day FIRE_NUMBER_THRESHOLD_FIRST_DAY] [--measurement_threshold_first_day MEASUREMENT_THRESHOLD_FIRST_DAY]
                              [--fire_number_threshold_second_day FIRE_NUMBER_THRESHOLD_SECOND_DAY] [--measurement_threshold_second_day MEASUREMENT_THRESHOLD_SECOND_DAY]
                              input_path output_file meteo_variables [meteo_variables ...]

Create a training set from the postprocessed data.

positional arguments:
  input_path            Path to the postprocessed data file or directory of files thatare concatenated.
  output_file           Path to the output file.
  meteo_variables       List of meteo variables to include in the training set.

options:
  -h, --help            show this help message and exit
  --filename_start FILENAME_START
                        File name start.
  --filter_nans FILTER_NANS
                        Filter out nan values.
  --fire_number_threshold_first_day FIRE_NUMBER_THRESHOLD_FIRST_DAY
                        Data with less fire occurencies on the first day are filtered out.
  --measurement_threshold_first_day MEASUREMENT_THRESHOLD_FIRST_DAY
                        Data with less measurements on the first day are filtered out.
  --fire_number_threshold_second_day FIRE_NUMBER_THRESHOLD_SECOND_DAY
                        Data with less fire occurencies on the second day are filtered out.
  --measurement_threshold_second_day MEASUREMENT_THRESHOLD_SECOND_DAY
                        Data with less measurements on the second day are filtered out.
```

### 2. Deep learning
Deep learning tools are found in the module `fire_forecast.deep_learning`. Here, you find the desciption of the main parts of the module and how to use them.
#### 2.1 Training a new model
New models can be either trained by the script `fire_forecast.deep_learning.train` or using the `Iterator` object of `fire_forecast.deep_learning.iterator`. The model as well as the training are defined via a config file in a `yaml` format. Examples of these config files with descriptions of the parameters are found in the directory `fire_forecast/deep_learning/configs`.

**By script:**
The script `fire_forecast.deep_learning.train` takes a config file as input and trains a model. The usage is as follows:
```
usage: train.py [-h] config

positional arguments:
  config      Path to the config file.

options:
  -h, --help  show this help message and exit
```

**By iterator:**
Use the iterator as follows:
```python
from fire_forecast.deep_learning.iterator import Iterator #Import the module
from fire_forecast.utils import read_config #Import the function to read the config file

config = read_config("path/to/config.yaml") #Read the config file
iterator = Iterator(config) #Create an iterator object
iterator.train() #Train the model
```

#### 2.2 Initializing/loading a model
The easiest way to initialize/load a model is via the `load_model_from_config` function of `fire_forecast.deep_learning.models`:
```python
from fire_forecast.deep_learning.models import load_model_from_config
from fire_forecast.deep_learning.utils import read_config

config = read_config("path/to/config.yaml") #Read the config file
model = load_model_from_config(config["model"]) #Load the model from the config part for the model
```
In the config file, there is the option to load the parameters of a pretrained model which will then automatically be done.


Alternatively, to initialize the model directly import the respective model. E.g. ofr the `ResidualNetwork` for the parameters see the docstring of the models:
```python
from fire_forecast.deep_learning.models import ResidualNetwork
model = ResidualNetwork(...)
```

#### 2.3 Loading a dataset
To load a dataset the `FireDataset` class of `fire_forecast.deep_learning.firedataset` can be used. 
It is recommended to use the `only_center=True` option, which will only load the center pixel of the 3x3 grid of the timeseries. This is the default option. An item of the `FireDataset` consists of 3 parts:
 * `fire_features`: 24 hourly fire radiative power of the previous day and the rate of measurement of the current day
 * `meteo_features`: 24 hourly meteorological data of the previous day and the current day 
 * `labels`: 24 hourly fire radiative power of the current day and the rate of measurement of the current day

To predict with the model the data first needs to be put into the correct shape. This can be done with the `flatten_features` function of `fire_forecast.deep_learning.utils`:
```python
from fire_forecast.deep_learning.firedataset import FireDataset
from fire_forecast.deep_learning.utils import flatten_features

dataset = FireDataset("path/to/dataset.hdf", only_center=True)

fire_features, meteo_features, labels = dataset[0]
input_features_in_correct_shape = flatten_features(fire_features, meteo_features)
```

#### 2.4 Predicting with a model
The full path to predict with a model is:
```python
from fire_forecast.deep_learning.models import load_model_from_config
from fire_forecast.deep_learning.utils import read_config, flatten_features
from fire_forecast.deep_learning.firedataset import FireDataset
import torch

config = read_config("path/to/config.yaml") #Read the config file
model = load_model_from_config(config["model"]) #Load the model from the config part for the model
dataset = FireDataset("path/to/dataset.hdf", only_center=True) #Load the dataset

fire_features, meteo_features, labels = dataset[0] #Get the first item of the dataset
input_features_in_correct_shape = flatten_features(fire_features, meteo_features) #Put the features into the correct shape
prediction = model(torch.from_numpy(input_features_in_correct_shape)) #Predict with the model
```
Note that this assumes that the model and the data are on the same "device". If the model is on a GPU use `.to(model.device)` on the input.
#### 2.5 Predicting with multiple models
To predict with multiple models, you can use the `Ensemble` class of `fire_forecast.deep_learning.ensemble`:
```python
from fire_forecast.deep_learning.ensemble import Ensemble
from fire_forecast.deep_learning.fire_dataset import FireDataset

configs = [
    "path/to/config1.yaml",
    "path/to/config2.yaml",
    "path/to/config3.yaml",
]
ensemble = Ensemble(configs) #Load the ensemble of models
dataset = FireDataset("path/to/dataset.hdf", only_center=True) #Load the dataset

fire_features, meteo_features, labels = dataset[0] #Get the first item of the dataset
ensemble_mean, ensemble_std = ensemble.predict(fire_features, meteo_features) #Predict with the ensemble
```

### 3. Evaluation
This code contains a set of functions and methods for evaluating the performance of forecasting models. The code is written in Python and uses several libraries such as NumPy, xarray, Matplotlib, Pandas, and Scikit-learn.

#### 3.1 Functions

##### `_check_dims(y_true, y_pred, weights=None)`

This function checks the dimensions of the input arrays and reshapes them if necessary. It returns the reshaped arrays.

##### `mae(y_true, y_pred, weights=None, keepdims=False)`

This function calculates the mean absolute error between the true and predicted values. It takes the following arguments:

- `y_true`: An array-like object containing the true values.
- `y_pred`: An array-like object containing the predicted values.
- `weights`: An array-like object containing the weights for each sample.
- `keepdims`: A boolean indicating whether to keep the dimensions of the target variable.

##### `rmse(y_true, y_pred, weights=None, keepdims=False)`

This function calculates the root mean squared error between the true and predicted values. It takes the following arguments:

- `y_true`: An array-like object containing the true values.
- `y_pred`: An array-like object containing the predicted values.
- `weights`: An array-like object containing the weights for each sample.
- `keepdims`: A boolean indicating whether to keep the dimensions of the target variable.

##### `correlation(y_true, y_pred, keepdims=False)`

This function calculates the correlation between the true and predicted values. It takes the following arguments:

- `y_true`: An array-like object containing the true values.
- `y_pred`: An array-like object containing the predicted values.
- `keepdims`: A boolean indicating whether to keep the dimensions of the target variable.

##### `train_models(X, y)`

This function trains a set of models on the given data. It returns a dictionary containing the trained models. It takes the following arguments:

- `X`: An array-like object containing the features.
- `y`: An array-like object containing the target variable.

##### `predict(models, X)`

This function predicts the target variable using the given models. It returns a dictionary containing the predictions. It takes the following arguments:

- `models`: A dictionary containing the trained models.
- `X`: An array-like object containing the features.

##### `evaluate_models(models, X, y_true, predictions=None, weights=None)`

This function compares the predictions of the given models to the true values. It returns a pandas DataFrame containing the metrics for each model. It takes the following arguments:

- `models`: A dictionary containing the trained models.
- `X`: An array-like object containing the features.
- `y_true`: An array-like object containing the true values.
- `predictions`: A dictionary containing the predictions.
- `weights`: An array-like object containing the weights for each sample.

#### 3.2 Example

```python
from fire_forecast.deep_learning import Iterator
from fire_forecast.forecasting import evaluation

# Iterator from deep_learning module
iterator: Iterator = ...


# Training

fire_features, meteo_features, labels = iterator.train_dataset[:]
features = flatten_features(fire_features, meteo_features)
target_values, weights = flatten_labels_and_weights(labels)

# The training features with shape (n_samples, n_features)
X: np.array = features
# The training targets with shape (n_samples, n_targets)
y: np.array = target_values

models = evaluation.train_models(X, y)

# Testing

fire_features, meteo_features, labels = iterator.test_dataset[:]
features = flatten_features(fire_features, meteo_features)
target_values, weights = flatten_labels_and_weights(labels)

with torch.no_grad():
    predictions = model(torch.from_numpy(features))
    predictions = predictions.numpy()

# The test features with shape (n_samples, n_features)
X_test: np.array = features
# The test targets with shape (n_samples, n_targets)
y_test: np.array = target_values
# The predictions of other models as a dictionary of model name to y_pred with shape (n_samples, n_targets)
predictions: dict = {
    "NN": predictions,
    "Persistence": fire_features,
}
# The weights of the labels as np.array with shape (n_samples, n_targets)
weights_test: np.array = weights

metrics = evaluation.evaluate_models(
    models=models,
    X=X_test,
    y=y_test,
    predictions=predictions,
    weights=weights_test,
)

metrics.head()
```
