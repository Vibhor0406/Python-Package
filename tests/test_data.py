from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from mluno.data import make_line_data,make_sine_data, split_data
from mluno.regressors import KNNRegressor, LinearRegressor
import numpy as np

def test_make_line_data_default():
    X, y = make_line_data()
    assert len(X) == 100
    assert len(y) == 100
    assert X.shape == (100, 1)
    assert np.issubdtype(X.dtype, np.floating)
    assert np.issubdtype(y.dtype, np.floating)

def test_make_line_data_custom():
    X, y = make_line_data(n_samples=200, beta_0=2, beta_1=3, sd=4, X_low=-5, X_high=5, random_seed=1)
    assert len(X) == 200
    assert len(y) == 200
    assert X.shape == (200, 1)
    assert np.issubdtype(X.dtype, np.floating)
    assert np.issubdtype(y.dtype, np.floating)
    assert np.min(X) >= -5
    assert np.max(X) <= 5

def test_make_line_data_with_seed():
    X1, y1 = make_line_data(n_samples=200, beta_0=2, beta_1=3, sd=4, X_low=-5, X_high=5, random_seed=42)
    X2, y2 = make_line_data(n_samples=200, beta_0=2, beta_1=3, sd=4, X_low=-5, X_high=5, random_seed=42)
    X3, y3 = make_line_data(n_samples=200, beta_0=2, beta_1=3, sd=4, X_low=-5, X_high=5, random_seed=1)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert not np.array_equal(X1, X3)
    assert not np.array_equal(y1, y3)

def test_make_line_data_without_seed():
    X1, y1 = make_line_data(n_samples=200, beta_0=2, beta_1=3, sd=4, X_low=-5, X_high=5)
    X2, y2 = make_line_data(n_samples=200, beta_0=2, beta_1=3, sd=4, X_low=-5, X_high=5)
    assert not np.array_equal(X1, X2)
    assert not np.array_equal(y1, y2)

def test_make_sine_data_default():
    X, y = make_sine_data()
    assert len(X) == 100
    assert len(y) == 100
    assert X.shape == (100, 1)
    assert np.issubdtype(X.dtype, np.floating)
    assert np.issubdtype(y.dtype, np.floating)

def test_make_sine_data_custom():
    X, y = make_sine_data(n_samples=200, sd=4, X_low=-5, X_high=5, random_seed=1)
    assert len(X) == 200
    assert len(y) == 200
    assert X.shape == (200, 1)
    assert np.issubdtype(X.dtype, np.floating)
    assert np.issubdtype(y.dtype, np.floating)
    assert np.min(X) >= -5
    assert np.max(X) <= 5

def test_make_sine_data_with_seed():
    X1, y1 = make_sine_data(random_seed=42)
    X2, y2 = make_sine_data(random_seed=42)
    X3, y3 = make_sine_data(random_seed=1)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert not np.array_equal(X1, X3)
    assert not np.array_equal(y1, y3)

def test_make_sine_data_without_seed():
    X1, y1 = make_sine_data(n_samples=100)
    X2, y2 = make_sine_data(n_samples=100)
    assert not np.array_equal(X1, X2)
    assert not np.array_equal(y1, y2)

def test_split_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    X_train, X_test, y_train, y_test = split_data(X, y, holdout_size=0.2, random_seed=42)

    assert X_train.shape[0] == 4
    assert X_test.shape[0] == 1
    assert y_train.shape[0] == 4
    assert y_test.shape[0] == 1

def test_split_data_with_seed():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_seed=42)
    X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_seed=42)
    X_train3, X_test3, y_train3, y_test3 = split_data(X, y, random_seed=1)

    assert np.array_equal(X_train1, X_train2)
    assert np.array_equal(X_test1, X_test2)
    assert np.array_equal(y_train1, y_train2)
    assert np.array_equal(y_test1, y_test2)
    assert not np.array_equal(y_train1, y_train3)
    assert not np.array_equal(X_train1, X_train3)