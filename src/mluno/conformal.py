import numpy as np

class ConformalPredictor:
    """
    A class used to represent a (Split) Conformal Predictor.
    $$ \\hat{C}_\\alpha(x) = \\left[ \\hat{u}(x) \\pm q_{1-\\alpha}(S) \\right]
    $$

    Parameters
    ----------
    regressor : object
        A regressor object that has a 'predict' method.
    alpha : float, optional
        The significance level used in the prediction interval calculation, by default 0.05.

    Attributes
    ----------
    scores : ndarray
        The conformity scores of the calibration data.
    quantile : float
        The empirical quantile of the conformity scores.
    """

    def __init__(self, regressor, alpha=0.05):
        """
        Constructs all the necessary attributes for the ConformalPredictor object.

        Parameters
        ----------
        regressor : object
            A regressor object that has a 'predict' method.
        alpha : float, optional
            The significance level used in the prediction interval calculation, by default 0.05.
        """
        self.regressor = regressor
        self.alpha = alpha
        self.scores = None
        self.quantile = None

    def fit(self, X, y):
        """
        Calibrates the conformal predictor using the provided calibration set.

        Specifically, the fit method learns
        $$ {q_{1-\\alpha}}(S) $$
        
        where ${q_{1-\\alpha}}(S)$ is the $(1 - {\\alpha})$ empirical quantile of the conformity scores

        $$  S = \\left\\{ |y_i - \\hat{u}(x_i)| \\right\\} \\cup \\{ \\infty \\} 
        $$


       
        Parameters
        ----------
        X : ndarray
            The input data for calibration.
        y : ndarray
            The output data for calibration.
        """
        # Fit the regressor
        self.regressor.fit(X, y)

        # Predict on calibration set
        preds_calib = self.regressor.predict(X)

        # Compute conformity scores (absolute residuals)
        self.scores = np.abs(preds_calib - y)

        # Compute the quantile
        self.quantile = np.quantile(self.scores, 1 - self.alpha)

    def predict(self, X):
        """
        Predicts the output for the given input X and provides a prediction interval.

        $$ \\hat{C}_\\alpha(x) = \\left[ \\hat{u}(x) \\pm q_{1-\\alpha}(S) \\right]
        $$

        Parameters
        ----------
        X : ndarray
            The input data for which to predict the output.

        Returns
        -------
        tuple
            A tuple containing the prediction (1D ndarray) and the lower (1D ndarray) and upper bounds (1D ndarray) of the prediction interval.
        """
        # Predict on new data
        preds_new = self.regressor.predict(X)

        # Construct prediction intervals
        lower_bounds = preds_new - self.quantile
        upper_bounds = preds_new + self.quantile

        return preds_new, lower_bounds, upper_bounds
