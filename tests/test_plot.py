from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from mluno.data import make_line_data,make_sine_data, split_data
from mluno.regressors import KNNRegressor, LinearRegressor
from mluno.plot import plot_predictions, plt
import numpy as np
from mluno.conformal import ConformalPredictor

def test_plot_predictions():
    X, y = make_line_data()
    regressor = LinearRegression()
    regressor.fit(X, y)

    fig, ax = plot_predictions(X, y, regressor, title="Test Plot")

    # Check that the function returns a Figure and Axes object
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Check that the title of the plot is correct
    assert ax.get_title() == "Test Plot"

def test_plot_predictions_conformal():
    X, y = make_line_data()
    regressor = LinearRegression()
    regressor.fit(X, y)
    conformal = ConformalPredictor(regressor)
    conformal.fit(X, y)

    fig, ax = plot_predictions(X, y, conformal, conformal=True, title="Test Plot Conformal")

    # Check that the function returns a Figure and Axes object
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Check that the title of the plot is correct
    assert ax.get_title() == "Test Plot Conformal"