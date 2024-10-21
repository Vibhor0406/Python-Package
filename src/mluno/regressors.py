import numpy as np

class KNNRegressor:
    """
    A class used to represent a K-Nearest Neighbors Regressor.
  
    Parameters
    ----------
    k : int, optional
        The number of nearest neighbors to consider for regression, by default 5.

    """

    def __init__(self, k=5):
        """
        Constructs all the necessary attributes for the KNNRegressor object.

        Parameters
        ----------
        k : int, optional
            The number of nearest neighbors to consider for regression, by default 5.
        """
        self.k = k

    def fit(self, X, y):
        """
        Fit the model using X as input data and y as target values.

        Parameters
        ----------
        X : ndarray
            The training data, which is a 2D array of shape (n_samples, 1) where each row is a sample and each column is a feature.
        y : ndarray
            The target values, which is a 1D array of shape (n_samples, ).
        """
        self.X = X
        self.y = y

    def predict(self, X_new):
        """
        Predict the target for the provided data.

        Parameters
        ----------
        X_new : ndarray
            Input data, a 2D array of shape (n_samples, 1), with which to make predictions.

        Returns
        -------
        ndarray
            The target values, which is a 1D array of shape (n_samples, ).
        """
        predicted_labels = [self._predict(x) for x in X_new]
        return np.array(predicted_labels)

    def _predict(self, x_new):
        # compute distances between new x and all samples in the X data
        distances = [np.linalg.norm(x_new - x) for x in self.X]
        # sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[: self.k]
        # extract the labels of the k nearest neighbor training samples
        k_nearest_y = self.y[k_indices]
        # return the mean of the k nearest neighbors
        return np.mean(k_nearest_y)

class LinearRegressor:

    """
    A class used to represent a Simple Linear Regressor.
    
    $$ Y = \\beta_0 + \\beta_1 \\cdot x + \\epsilon 
    $$

    Attributes
    ----------
    weights : ndarray

        The weights of the linear regression model. Here, the weights are represented by the ${\\beta}$
        vector which for univariate regression is a 1D vector of length two, ${\\beta = [\\beta_0,\\beta_1]}$ , where ${\\beta_0}$ is the slope and ${\\beta_1}$ is the intercept.
    """

    def __init__(self):
        self.weights = None

    def __repr__(self) -> str:
        return f"Linear Regression model with weights = {self.weights}."

    def fit(self, X, y):
        """
        Trains the linear regression model using the given training data. In other words, the fit method learns the weights, represented by the ${\\beta}$ vector. To learn the ${\\beta}$ vector, use:
        $${\\hat\\beta  = (X^TX)^{-1}X^Ty}$$

        Here, $X$ is the so-called design matrix, which, to include a term for the intercept, has a column of ones appended to the input X matrix.

        Parameters
        ----------
        X : ndarray
            The training data, which is a 2D array of shape (n_samples, 1) where each row is a sample and each column is a feature.
        y : ndarray
            The target values, which is a 1D array of shape (n_samples, ).

        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X):
        """
        Makes predictions for input data.

        $$ \\hat y = X\\hat\\beta
        $$

        Parameters
        ----------
        X : ndarray
            Input data, a 2D array of shape (n_samples, 1), with which to make predictions.

        Returns
        -------
        ndarray
            The predicted target values as a 1D array with the same length as X.
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.weights

