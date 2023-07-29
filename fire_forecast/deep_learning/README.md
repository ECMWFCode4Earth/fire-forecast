# Deep Learning
This directory contains the building blocks of the training of neural networks. Preliminary you shoud have data that is produced according to the README in the fodler data_preparation.

## Train a model
To train a model you need a config that describes the training process.
Exapmles for configs can be found in the configs folder.

The training can then be started with:
```
from fire_forecast.deep_learning.models import load_model_from_config
from fire_forecast.deep_learning.iterator import Iterator
from fire_forecast.deep_learning.utils import read_config
import yaml

config = read_config("/home/chlw/software/repositories/fire-forecast/fire_forecast/deep_learning/configs/example0.yaml")
iterator = Iterator(config)

iterator.train()
```
This will start a training according to the settings in the config.
A copy of the config is also saved to the output folder so a model can be reproduced, and checkpoints of the networks can be found in the output folder to continue a training.

To find out the input size:
```
from fire_forecast.deep_learning.fire_dataset import FireDataset
x = FireDataset("train.hdf")
x.input_size
```