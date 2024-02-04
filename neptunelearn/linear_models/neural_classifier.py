"""Implements the Perceptron & Adaline Learning Algorithm
Author: Rajan Subramanian

Note: for the perceptron algorithm, if weights are initialized to 0, 
learning rate eta has no effect on decision boundary
- so initializing weights to 0 affects only the scale of weights not direction
- todo: need to either correctly name fit_online or remove this as i dont know where this came from
"""

from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neptunelearn.linear_models.base import NeuralBase


class Perceptron(NeuralBase):
    """Implements the Perceptron Learning Algorithm"""

    def __init__(self, eta: float = 0.01, bias: bool = True, tol=10e-6):
        self.eta = eta
        self.bias = bias
        self.thetas = None
        self.weights = None
        self.degree = 1
        self.tol = tol

    def _get_learner(self, learner: str) -> Callable[[NDArray, NDArray], Perceptron]:
        if learner == "batch":
            return self._fit_batch
        elif learner == "online":
            return self._fit_online
        else:
            return ValueError(learner)

    def _fit_batch(self, X: NDArray, y: NDArray) -> Perceptron:
        """fits training data using batch gradient descent

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, d_features)
            n_samples is number of instances i.e rows
            d_features is number of features (dimension of data)
        y : np.ndarray
            target variable

        Returns
        -------
        Perception
            object with fitted parameters
        """
        # Add bias unit to design matrix
        degree, bias = self.degree, self.bias
        X = self.make_polynomial(X, degree, bias)

        # Initialize weights to 0
        self.thetas = np.zeros(X.shape[1])
        weights = {}
        index = -1
        converged = False
        self.errors = []
        while not converged:
            error = 0
            index += 1
            prev_weights = self.thetas.copy()
            # for each example in training set
            for xi, target in zip(X, y):
                # update weights if there are misclassifications
                if target != self.predict(xi):
                    self.thetas += target * xi
                    error += 1
            self.errors.append(error)

            weights[index] = self.thetas.copy()
            if np.linalg.norm(self.thetas - prev_weights) <= self.tol:
                converged = True

        self.weights = pd.DataFrame.from_dict(
            weights, orient="index", columns=["bias", "weight1", "weight2"]
        )
        return self

    def _fit_online(self, X: NDArray, y: NDArray) -> Perceptron:
        R = (np.sum(np.abs(X) ** 2, axis=-1) ** (0.5)).max()
        bias = 0
        # Initialize weights to 0
        self.thetas = np.zeros(X.shape[1] + 1)
        mistakes = 0
        mistakes_from_previous_iteration = 0
        while True:
            for xi, target in zip(X, y):
                if target * (self.net_input(xi, self.thetas[1:]) + bias) <= 0:
                    self.thetas[1:] += self.eta * target * xi
                    bias += self.eta * target * R * R
                    mistakes += 1
            if mistakes_from_previous_iteration == mistakes:
                break
            mistakes_from_previous_iteration = mistakes

        self.thetas[0] = bias
        return self

    def fit(self, X: NDArray, y: NDArray, learner="batch") -> Perceptron:
        learner = self._get_learner(learner)
        return learner(X, y)

    def predict(self, X: NDArray) -> int:
        """Activation function to determine if neuron should fire or not

        Parameters
        ----------
        X : np.ndarray
            design matrix that includes the bias
        thetas : Union[np.ndarray, None], optional
            weights from fitting, by default None

        Returns
        -------
        np.ndarray
            predictions
        """
        return 1 if self.net_input(X, self.thetas) >= 0 else -1

    def plot_decision_boundary(self, inputs, targets, weights):
        for input, target in zip(inputs, targets):
            plt.plot(input[0], input[1], "ro" if (target == 1.0) else "bo")

        slope = -weights[1] / weights[2]
        intercept = -weights[0] / weights[2]
        for i in np.linspace(np.amin(inputs[:, :1]), np.amax(inputs[:, :1])):
            y = (slope * i) + intercept
            plt.plot(i, y, "ko")

    def plot_misclassification_errors(self):
        plt.plot(range(1, len(self.errors) + 1), self.errors, marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Number of updates")
        plt.grid()


class Adaline(NeuralBase):
    """Implements the Adaptive Linear Neuron by Bernard Widrow
    via Batch Gradient Descent

    Notes:
        - The cost function is given by J(w) = 1/2 ||(yi - yhat)||
    """

    def __init__(self, eta: float = 0.01, niter: int = 50, bias: bool = True):
        """Default Constructor used to initialize the Adaline model"""
        self.eta = eta
        self.niter = niter
        self.cost = []
        self.bias = bias
        self.thetas = None
        self.degree = 1

    def fit(self, X: NDArray, y: NDArray) -> Adaline:
        """fits training data via batch gradient descent

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, p_features)
            n_samples is number of instances i.e rows
            p_features is number of features (dimension of data)
        y : np.ndarray
            response variable

        Returns
        -------
        Perception
            object with fitted parameters
        """
        # add bias + weights for each neuron
        self.thetas = np.zeros(shape=1 + X.shape[1])
        n = X.shape[0]
        # Add bias unit to design matrix
        degree = self.degree
        X = self.make_polynomial(X, degree=degree, bias=True)

        for _ in range(self.niter):
            net_input = self.net_input(X, self.thetas)
            error = y - self.activation(net_input)
            loss = (error.T @ error) / (2.0 * n)
            self.thetas += self.eta * X.T @ error / X.shape[0]
            self.cost.append(loss)
        return self

    def plot_misclassification_errors(self, use_log: bool = True):
        if use_log:
            plt.plot(range(1, len(self.cost) + 1), np.log10(self.cost), marker="o")
            plt.ylabel("Log(Mean Squared Error)")
        else:
            plt.plot(range(1, len(self.cost) + 1), self.cost, marker="o")
            plt.ylabel("Mean Squared Error")
        plt.xlabel("Epochs")
        plt.grid()

    def activation(self, X: NDArray) -> NDArray:
        """Computes the linear activation function
        given by f(w'x) = w'x

        Parameters
        ----------
        X : np.ndarray
            output from the netinput function

        Returns
        -------
        np.ndarray
            activation function
        """
        return X

    def predict(self, X: NDArray, thetas: Optional[NDArray]) -> float:
        """Computes the class label after activation

        Parameters
        ----------
        X : np.ndarray
            [description]
        thetas : Union[np.ndarray, None], optional
            [description], by default None
        """
        if thetas is None:
            return 1 if self.activation(self.net_input(X, self.thetas)) >= 0.0 else -1
        else:
            return 1 if self.activation(self.net_input(X, thetas)) >= 0.0 else -1
