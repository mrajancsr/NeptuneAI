"""Abstract base class for regression models"""

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import PolynomialFeatures


class LinearBase(metaclass=ABCMeta):
    """Abstract Base class representing the Linear Model"""

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def make_polynomial(
        self, X: np.ndarray, degree: int, bias: bool
    ) -> NDArray:  # noqa
        pf = PolynomialFeatures(degree=degree, include_bias=bias)
        return pf.fit_transform(X)


class LogisticBase(LinearBase):
    """Abstract Base class representing a Logistic Regression Model"""

    def sigmoid(self, z: float) -> float:
        """Computes the sigmoid function

        Parameters
        ----------
        z : np.ndarray
            input value from linear transformation

        Returns
        -------
        np.ndarray
            sigmoid function value
        """
        return 1.0 / (1 + np.exp(-z))

    def net_input(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Computes the Linear transformation X@theta

        Parameters
        ----------
        X : np.ndarray, shape={n_samples, p_features}
            feature matrix
        theta : np.ndarray, shape={p_features + intercept}
            weights of logistic regression

        Returns
        -------
        np.ndarray
            linear transformation
        """
        return X @ theta


class NeuralBase(LinearBase):
    """Abstract Base class representing a Neural Network"""

    def net_input(self, X: np.ndarray, theta: np.ndarray) -> float:
        """Computes the net input vector
        z = w1x1 + w2x2 + ... + wpxp := w'x

        Parameters
        ----------
        X : np.ndarray, shape={n_samples, p_features}
            design matrix
        thetas : np.ndarray, shape={p_features + intercept}
            weights of neural classifier, w vector above
            assumes first element is the bias unit i.e intercept

        Returns
        -------
        np.ndarray
            linear transformation
        """
        return X @ theta
