from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from mluno.data import make_line_data,make_sine_data, split_data
from mluno.regressors import KNNRegressor, LinearRegressor
from mluno.plot import plot_predictions, plt
import numpy as np
from mluno.conformal import ConformalPredictor
from mluno.metrics import coverage

def test_conformal_predictor():
    # setup
    X_train, y_train = make_line_data(n_samples=200, random_seed=42)
    X_calibration, y_calibration = make_line_data(n_samples=1000, random_seed=42)
    X_test, y_test = make_line_data(n_samples=1000, random_seed=42)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    conformal_predictor = ConformalPredictor(regressor)
    conformal_predictor.fit(X_calibration, y_calibration)

    # test fit method
    assert conformal_predictor.scores is not None
    assert conformal_predictor.quantile is not None

    # test predict method
    y_pred, y_lower, y_upper = conformal_predictor.predict(X_test)
    assert np.all(y_pred == regressor.predict(X_test))
    assert np.all(y_lower <= y_pred)
    assert np.all(y_upper >= y_pred)

    # check coverage
    assert np.abs(coverage(y_test, y_lower, y_upper) - 0.95) < 0.025

    # check coverage with non-default alpha
    conformal_predictor = ConformalPredictor(regressor, alpha=0.2)
    conformal_predictor.fit(X_calibration, y_calibration)
    y_pred, y_lower, y_upper = conformal_predictor.predict(X_test)
    assert np.abs(coverage(y_test, y_lower, y_upper) - (1 - 0.20)) < 0.025