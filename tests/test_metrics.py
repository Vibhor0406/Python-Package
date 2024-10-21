
from mluno.metrics import rmse, mae, coverage, sharpness
import numpy as np

def test_rmse():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    assert rmse(y_true, y_pred) == 0

    y_pred = np.array([2, 3, 4, 5, 6])
    assert rmse(y_true, y_pred) == 1

def test_mae():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    assert mae(y_true, y_pred) == 0

    y_pred = np.array([2, 3, 4, 5, 6])
    assert mae(y_true, y_pred) == 1

def test_coverage():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred_lower = np.array([0, 1, 2, 3, 4])
    y_pred_upper = np.array([2, 3, 4, 5, 6])
    assert coverage(y_true, y_pred_lower, y_pred_upper) == 1

    y_pred_upper = np.array([1, 2, 3, 3, 3])
    assert coverage(y_true, y_pred_lower, y_pred_upper) == 0.6

def test_sharpness():
    y_pred_lower = np.array([0, 1, 2, 3, 4])
    y_pred_upper = np.array([2, 3, 4, 5, 6])
    assert sharpness(y_pred_lower, y_pred_upper) == 2

    y_pred_upper = np.array([1, 2, 3, 4, 5])
    assert sharpness(y_pred_lower, y_pred_upper) == 1