from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from mluno.data import make_line_data,make_sine_data, split_data
from mluno.regressors import KNNRegressor, LinearRegressor
import numpy as np

def test_knn_regressor():
    # linear data
    for seed in [1, 2, 3, 4, 5]:
        X, y = make_line_data(random_seed=seed)
        # use default k
        knn = KNNRegressor()
        knn_sk = KNeighborsRegressor()
        knn.fit(X, y)
        knn_sk.fit(X, y)
        assert np.all(knn.predict(X) == knn_sk.predict(X))
        # use non-default k
        knn = KNNRegressor(k=10)
        knn_sk = KNeighborsRegressor(n_neighbors=10)
        knn.fit(X, y)
        knn_sk.fit(X, y)
        assert np.all(knn.predict(X) == knn_sk.predict(X))
    # sine data data
    for seed in [1, 2, 3, 4, 5]:
        X, y = make_sine_data(random_seed=seed)
        # use default k
        knn = KNNRegressor()
        knn_sk = KNeighborsRegressor()
        knn.fit(X, y)
        knn_sk.fit(X, y)
        assert np.all(knn.predict(X) == knn_sk.predict(X))
        # use non-default k
        knn = KNNRegressor(k=10)
        knn_sk = KNeighborsRegressor(n_neighbors=10)
        knn.fit(X, y)
        knn_sk.fit(X, y)
        assert np.all(knn.predict(X) == knn_sk.predict(X))

def test_linear_regressor():
    for seed in [1, 2, 3, 4, 5]:
        X, y = make_line_data(random_seed=seed)
        lr = LinearRegressor()
        lr_sk = LinearRegression()
        lr.fit(X, y)
        lr_sk.fit(X, y)
        assert np.sum(np.abs(lr.predict(X) - lr_sk.predict(X))) < 0.01



