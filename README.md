# fire-forecast
Tool for forecast of fire radiative power.

## Contribution
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
 1. `fire_forecast.data_preparation` for preprocessing data into various forms
 2. `fire_forecast.deep_learning` for creating, training and loading deep learning models
 3. `fire_forecast.fire_model` for classical machine learnign approaches
In the following we will give a short overview of the different parts and how to use them.

All scripts can be called with python -m fire_forecast.module_name.script_name -h to get a help message.

### 1. Data preparation
The preprocessing scripts are assuing gridded data in an `netcdf` forma that includes dataarrays for `frpfire` and `offire`. The models are designed, to take input in form of a fire timeseries, which are produced by the preprocessing tools.
#### 1.1. Selecting timeseries of pixels with fire
#TODO: fill here

The resulting timeseries can be used for the classical approches in `#TODO: fill here`.
#### 1.2. Cutting timeseries into 2 day samples for the deep learning models
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
#### 1.3 Converting the data to h5 format
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
New models can be either trained by the script `fire_forecast.deep_learning.train` or using the `Iterator` object of `fire_forecast.deep_learning.iteratior`. The model as well as the training are defined via a config file in a `yaml` format. Examples of these config files with descriptions of the parameters are found in the directory `fire_forecast/deep_learning/configs`.

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
```
from fire_forecast.deep_learning.iterator import Iterator #Import the module
from fire_forecast.utils import read_config #Import the function to read the config file

config = read_config("path/to/config.yaml") #Read the config file
iterator = Iterator(config) #Create an iterator object
iterator.train() #Train the model
```

#### 2.2 Initializing/loading a model
The easiest way to initialize/load a model is via the `load_model_from_config` function of `fire_forecast.deep_learning.models`:
```
from fire_forecast.deep_learning.models import load_model_from_config
from fire_forecast.deep_learning.utils import read_config

config = read_config("path/to/config.yaml") #Read the config file
model = load_model_from_config(config["model"]) #Load the model from the config part for the model
```
In the config file, there is the option to load the parameters of a pretrained model which will then automatically be done.


Alternatively, to initialize the model directly import the respective model. E.g. ofr the `ResidualNetwork` for the parameters see the docstring of the models:
```
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
```
from fire_forecast.deep_learning.firedataset import FireDataset
from fire_forecast.deep_learning.utils import flatten_features

dataset = FireDataset("path/to/dataset.hdf", only_center=True)

fire_features, meteo_features, labels = dataset[0]
input_features_in_correct_shape = flatten_features(fire_features, meteo_features)
```

#### 2.4 Predicting with a model
The full path to predict with a model is:
```
from fire_forecast.deep_learning.models import load_model_from_config
from fire_forecast.deep_learning.utils import read_config, flatten_features
from fire_forecast.deep_learning.firedataset import FireDataset

config = read_config("path/to/config.yaml") #Read the config file
model = load_model_from_config(config["model"]) #Load the model from the config part for the model
dataset = FireDataset("path/to/dataset.hdf", only_center=True) #Load the dataset

fire_features, meteo_features, labels = dataset[0] #Get the first item of the dataset
input_features_in_correct_shape = flatten_features(fire_features, meteo_features) #Put the features into the correct shape
prediction = model(input_features_in_correct_shape) #Predict with the model
```



