import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline


def _check_dims(y_true, y_pred, weights=None):
    if y_true.ndim == 1:
        y_true = y_true.reshape((1, -1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((1, -1))

    if weights is not None:
        if weights.ndim == 1:
            weights = weights.reshape((1, -1))
    else:
        weights = np.ones(y_true.shape)    
    return y_true, y_pred, weights


def mae(y_true, y_pred, weights=None, keepdims=False):
    """
    Calculates the mean absolute error between the true and predicted values.

    Args:
        y_true: An array-like object containing the true values.
        y_pred: An array-like object containing the predicted values.
        weights: An array-like object containing the weights for each sample.
        keepdims: A boolean indicating whether to keep the dimensions of the target variable.
            If True, the output will have the same dimensions as the target variable.
            If False, the output will be a scalar value.

    Returns:
        The mean absolute error between the true and predicted values.
    """
    y_true, y_pred, weights = _check_dims(y_true, y_pred, weights)

    mae = np.mean(np.abs(y_true - y_pred) * weights, axis=0)
    if keepdims:
        return mae
    else:
        return np.mean(mae)


def rmse(y_true, y_pred, weights=None, keepdims=False):
    """
    Calculates the root mean squared error between the true and predicted values.

    Args:
        y_true: An array-like object containing the true values.
        y_pred: An array-like object containing the predicted values.
        weights: An array-like object containing the weights for each sample.
        keepdims: A boolean indicating whether to keep the dimensions of the target variable.
            If True, the output will have the same dimensions as the target variable.
            If False, the output will be a scalar value like sklearn.metrics.mean_squared_error
            with squared=False.

    Returns:
        The root mean squared error between the true and predicted values.
    """
    y_true, y_pred, weights = _check_dims(y_true, y_pred, weights)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2 * weights, axis=0))
    if keepdims:
        return rmse
    else:
        return np.mean(rmse)


def correlation(y_true, y_pred, keepdims=False):
    """
    Calculates the correlation between the true and predicted values.

    Args:
        y_true: An array-like object containing the true values.
        y_pred: An array-like object containing the predicted values.
        keepdims: A boolean indicating whether to keep the dimensions of the target variable.
            If True, the output will have the same dimensions as the target variable.
            If False, the output will be a scalar value.
    """
    y_true, y_pred, _ = _check_dims(y_true, y_pred)

    corr = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        corr[i] = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
    if keepdims:
        return corr
    else:
        return np.mean(corr)


def train_models(X, y):
    """
    Trains a set of models on the given data.

    Args:
        X: An array-like object containing the features.
        y: An array-like object containing the target variable.

    Returns:
        A dictionary containing the trained models.
    """
    models = {
        "linear": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("linear", LinearRegression()),
            ]
        ),
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1)),
            ]
        ),
        # "svm": Pipeline(
        #     [
        #         ("scaler", StandardScaler()),
        #         ("svm", svm.SVR()),
        #     ]
        # ),
        "tree": DecisionTreeRegressor(
            max_depth=3, min_samples_split=3, min_samples_leaf=10, 
        ),
    }
    for name, model in models.items():
        model.fit(X, y)
    return models


def predict(models, X):
    """
    Predicts the target variable using the given models.

    Args:
        models: A dictionary containing the trained models.
        X: An array-like object containing the features.

    Returns:
        A dictionary containing the predictions.
    """
    y_pred = {}
    for name, model in models.items():
        y_pred[name] = model.predict(X)
    return y_pred


def evaluate_models(models, X, y_true, predictions=None, weights=None):
    """
    Compares the predictions of the given models to the true values.

    Args:
        models: A dictionary containing the trained models.
        X: An array-like object containing the features.
        y_true: An array-like object containing the true values.
        predictions: A dictionary containing the predictions.
        weights: An array-like object containing the weights for each sample.

    Returns:
        A pandas DataFrame containing the metrics for each model.
    """
    y_pred = predict(models, X)
    metrics = {
        name: {
            "mae": mae(y_true, y_pred[name], weights),
            "rmse": rmse(y_true, y_pred[name], weights),
            "correlation": correlation(y_true, y_pred[name]),
        }
        for name in models.keys()
    }
    if predictions is not None:
        for name, pred in predictions.items():
            metrics[name] = {}
            metrics[name]["mae"] = mae(y_true, pred, weights)
            metrics[name]["rmse"] = rmse(y_true, pred, weights)
            metrics[name]["correlation"] = correlation(y_true, pred)
    df = pd.DataFrame(metrics).T
    return df
