import numpy as np

def rmse(y_true, y_pred):
    """
    Compute the Root Mean Square Error (RMSE).

    Parameters
    ----------
    y_true : ndarray
        A 1D array of the true target values.
    y_pred : ndarray
        A 1D array of the predicted target values.

    Returns
    -------
    float
        The RMSE between the true and predicted target values.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """
    Compute the Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : ndarray
        A 1D array of the true target values.
    y_pred : ndarray
        A 1D array of the predicted target values.

    Returns
    -------
    float
        The MAE between the true and predicted target values.
    """
    return np.mean(np.abs(y_true - y_pred))

def coverage(y_true, y_pred_lower, y_pred_upper):
    """
    Compute the coverage of the prediction intervals.

    Parameters
    ----------
    y_true : ndarray
        A 1D array of the true target values.
    y_pred_lower : ndarray
        A 1D array of lower bounds of the predicted intervals.
    y_pred_upper : ndarray
        A 1D array of upper bounds of the predicted intervals.

    Returns
    -------
    float
        The proportion of true values that fall within the predicted intervals.
    """
    return np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))

def sharpness(y_pred_lower, y_pred_upper):
    """
    Compute the sharpness of the prediction intervals.

    Parameters
    ----------
    y_pred_lower : ndarray
        A 1D array of lower bounds of the predicted intervals.
    y_pred_upper : ndarray
        A 1D array of upper bounds of the predicted intervals.

    Returns
    -------
    float
        The average width of the predicted intervals.
    """
    return np.mean(y_pred_upper - y_pred_lower)