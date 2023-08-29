import numpy as np
from fire_forecast.forecasting import evaluation
from sklearn.metrics import mean_squared_error


def test_mae():
    y_true = np.arange(20).reshape(10, 2)
    y_pred = np.arange(20).reshape(10, 2) + 1
    expected_mae = np.mean(np.abs(y_true - y_pred))
    assert np.allclose(evaluation.mae(y_true, y_pred), expected_mae)
    assert evaluation.mae(y_true, y_pred, keepdims=True).shape == (2,)

def test_rmse():
    y_true = np.arange(20).reshape(10, 2)
    y_pred = np.arange(20).reshape(10, 2) + 1
    expected_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    assert np.isclose(evaluation.rmse(y_true, y_pred), expected_rmse)
    assert evaluation.rmse(y_true, y_pred, keepdims=True).shape == (2,)


def test_correlation():
    y_true = np.arange(20).reshape(4, 5)
    y_pred = np.arange(20).reshape(4, 5) + 1
    assert evaluation.correlation(y_true, y_pred).shape == ()
    assert evaluation.correlation(y_true, y_pred, keepdims=True).shape == (5,)


def test_models():
    X = np.arange(20).reshape(10, 2)
    y = X @ np.array([1,2]) + 1
    weights = np.ones(y.shape)
    models = evaluation.train_models(X, y)
    predictions = evaluation.predict(models, X)
    for model, pred in predictions.items():
        assert pred.shape == y.shape
    compare = evaluation.evaluate_models(models, X, y)
    compare = evaluation.evaluate_models(models, X, y, weights)