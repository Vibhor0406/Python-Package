import numpy as np

def make_line_data(n_samples=100, beta_0=0, beta_1=1, sd=1, X_low=-10, X_high=10, random_seed=None):
    """
    Generate data for linear regression.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate, by default 100.
    beta_0 : int, optional
        The intercept of the line, by default 0.
    beta_1 : int, optional
        The slope of the line, by default 1.
    sd : int, optional
        Standard deviation of the normally distributed errors, by default 1.
    X_low : int, optional
        Lower bound for the uniform distribution from which X values are drawn, by default -10.
    X_high : int, optional
        Upper bound for the uniform distribution from which X values are drawn, by default 10.
    random_seed : int, optional
        The seed for the random number generator, by default None.

    Returns
    -------
    tuple
        A tuple containing the X and y arrays. X is a 2D array with shape (n_samples, 1) and y is a 1D array with shape (n_samples,). X contains the simulated X values and y contains the corresponding true mean of the linear model with added normally distributed errors.
    """
    
   
    if random_seed is not None:
        np.random.seed(random_seed)
    X = np.random.uniform(low=X_low, high=X_high, size=(n_samples, 1))
    y = beta_0 + beta_1 * X.ravel() + np.random.normal(scale=sd, size=n_samples)
    return X, y

    
def make_sine_data(n_samples=100, sd=1, X_low=-6, X_high=6, random_seed=None):
    """
    Generate data for nonlinear regression.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate, by default 100.
    sd : float, optional
        Standard deviation of the normally distributed errors, by default 1.
    X_low : float, optional
        Lower bound for simulated X values, by default -6.
    X_high : float, optional
        Upper bound for simulated X values, by default 6.
    random_seed : int, optional
        Seed to control randomness, by default None.

    Returns
    -------
    tuple
        A tuple containing the X and y arrays. X is a 2D array with shape (n_samples, 1) and y is a 1D array with shape (n_samples,). X contains the simulated X values and y contains the corresponding sine values with added normally distributed errors.
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    X = np.random.uniform(low=X_low, high=X_high, size=(n_samples, 1))
    y = np.sin(X.ravel()) + np.random.normal(scale=sd, size=n_samples)
    return X, y

def split_data(X, y, holdout_size=0.2, random_seed=None):
    """
    Split the data into train and test sets.

    Parameters
    ----------
    X : ndarray
        The feature data to be split. A 2D array with shape (n_samples, 1).
    y : ndarray
        The target data to be split. A 1D array with shape (n_samples,).
    holdout_size : float, optional
        The proportion of the data to be used as the test set, by default 0.2.
    random_seed : int, optional
        Seed to control randomness, by default None.

    Returns
    -------
    tuple
        The split train and test data: (X_train, X_test, y_train, y_test).
    """
    """
    Example
    -------
    >>> X, y = make_sine_data(n_samples=200, sd=0.5, X_low=-3, X_high=3, random_seed=42)
    >>> X_train, X_test, y_train, y_test = split_data(X, y, holdout_size=0.3, random_seed=42)
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    train_size = int((1 - holdout_size) * X.shape[0])
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test