import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(X, y, regressor, conformal=False, title=""):
    """
    Plot predictions of a regressor along with the data.
    Parameters
    ----------
    X : ndarray
        The input data to the regressor. A 2D array.
    y : ndarray
        The target values. A 1D array of the same length as X.
    regressor : object
        The regressor object. It should have a predict method that accepts X and returns predictions.
    conformal : bool
        If True, the regressor is assumed to return prediction intervals (lower and upper bounds) along with the predictions. The prediction intervals are plotted as a shaded area. Default is False.
    title : str
        The title of the plot. Default is an empty string.

    Returns
    -------

    fig : matplotlib$.$figure$.$Figure
        The figure object of the plot.
    ax : matplotlib$.$axes$.$Axes
        The axes object of the plot.       
    

    Notes
    -----
    This function assumes that the predict method of regressor returns a tuple of three elements (predictions, lower bounds, upper bounds) when conformal is True.
    """
    
    X_values = np.linspace(X.min(), X.max(), 500)[:, None]

    if conformal:
        y_pred, y_pred_lower, y_pred_upper = regressor.predict(X_values)
    else:
        y_pred = regressor.predict(X_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, label="Data", color="dodgerblue")
    ax.plot(X_values, y_pred, color="darkorange", label="Predicted")
    if conformal:
        ax.fill_between(
            X_values.flatten(),
            y_pred_lower,
            y_pred_upper,
            color="orange",
            alpha=0.2,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(color="grey", linestyle="--", linewidth=0.25, alpha=0.5)

    return fig, ax
