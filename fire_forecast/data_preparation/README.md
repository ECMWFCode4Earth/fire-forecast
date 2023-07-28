# Data Preparation
Here you find a set of scripts and the respective code base:
The scripts are:
 * `select_data.py` File to select 2 day training examples form the original gridded data format.
 * `select_data_all_shifts.py` Same as for the above but for 24 hourly shifts.
 * `cut_data.py` File to select 2 day training examples that are allready preprocessed into 3 by 3 grid.
 * `cut_data_all_shifts.py` Analogous to `select_data_all_shifts.py`
 * `create_training_set.py` Further filters data and saves it as h5 file for better usage with torch.

The help for each script can be called with:
```
python -m fire_forecast.data_preparation.script_name -h
```

## Example:
### 1. Split into samples
If there is data that is preprocessed into 3x3 slices first convert it to 2 day training samples, separatign them into train, test and validation datasets:
```
python -m fire_forecast.data_preparation.cut_data_all_shifts /data/input/0p5deg/longtimes/TestLongTimeSeries.nc --output_path ./TestLongTimeSeries.nc --validation_split 0.1 --test_split 0.1
```
This will produce 24*3 output files (1 per time shift and 1 per train/test/validation set) in the relative folder `./`.
each file will have a name similar to `TestLongTimeSeries_train_3.nc`.

### 2. Convert to h5 format
For converting to h5 you now can combine these 24*3 files to 3 sets for the training with:
```
python -m fire_forecast.data_preparation.create_training_set /directory_with_results_from_previous_steps ./train.hdf r tp --filename_start TestLongTimeSeries_train
```
and similar commands for validation and test set.

(For usage of data see README in deep_learning directory)