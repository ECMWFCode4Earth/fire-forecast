## Overview

This code contains a set of functions and methods for evaluating the performance of forecasting models. The code is written in Python and uses several libraries such as NumPy, xarray, Matplotlib, Pandas, and Scikit-learn.

## Functions

### `_check_dims(y_true, y_pred, weights=None)`

This function checks the dimensions of the input arrays and reshapes them if necessary. It returns the reshaped arrays.

### `mae(y_true, y_pred, weights=None, keepdims=False)`

This function calculates the mean absolute error between the true and predicted values. It takes the following arguments:

- `y_true`: An array-like object containing the true values.
- `y_pred`: An array-like object containing the predicted values.
- `weights`: An array-like object containing the weights for each sample.
- `keepdims`: A boolean indicating whether to keep the dimensions of the target variable.

### `rmse(y_true, y_pred, weights=None, keepdims=False)`

This function calculates the root mean squared error between the true and predicted values. It takes the following arguments:

- `y_true`: An array-like object containing the true values.
- `y_pred`: An array-like object containing the predicted values.
- `weights`: An array-like object containing the weights for each sample.
- `keepdims`: A boolean indicating whether to keep the dimensions of the target variable.

### `correlation(y_true, y_pred, keepdims=False)`

This function calculates the correlation between the true and predicted values. It takes the following arguments:

- `y_true`: An array-like object containing the true values.
- `y_pred`: An array-like object containing the predicted values.
- `keepdims`: A boolean indicating whether to keep the dimensions of the target variable.

### `train_models(X, y)`

This function trains a set of models on the given data. It returns a dictionary containing the trained models. It takes the following arguments:

- `X`: An array-like object containing the features.
- `y`: An array-like object containing the target variable.

### `predict(models, X)`

This function predicts the target variable using the given models. It returns a dictionary containing the predictions. It takes the following arguments:

- `models`: A dictionary containing the trained models.
- `X`: An array-like object containing the features.

### `evaluate_models(models, X, y_true, predictions=None, weights=None)`

This function compares the predictions of the given models to the true values. It returns a pandas DataFrame containing the metrics for each model. It takes the following arguments:

- `models`: A dictionary containing the trained models.
- `X`: An array-like object containing the features.
- `y_true`: An array-like object containing the true values.
- `predictions`: A dictionary containing the predictions.
- `weights`: An array-like object containing the weights for each sample.

## Example

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
    predictions = self.model(torch.from_numpy(features))
    predictions = predictions.numpy()

# The test features with shape (n_samples, n_features)
X_test: np.array = features
# The test targets with shape (n_samples, n_targets)
y_test: np.array = target_values
# The predictions of other models as a dictionary of model name to y_pred with shape (n_samples, n_targets)
predictions: dict = {"NN": predictions}
# The weights of the labels as np.array with shape (n_samples, n_targets)
weights_test: np.array = weights

metrics = evaluation.evaluate_models(
    models=models,
    X=X_test,
    y=y_test,
    predictions=predictions,
    weights=weights_test
)

metrics.head()
```