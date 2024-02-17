# pyre-strict
"""Implementation of Linear Regression using various fitting methods
Author: Rajan Subramanian
Created: May 23, 2020
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from numpy import log
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from neptunelearn.linear_models.base import LinearBase


@dataclass
class RegressionDiagnostics:
    X: NDArray
    y: NDArray
    theta: NDArray
    predictions: NDArray
    residuals: NDArray = field(
        init=False,
        default_factory=lambda: np.array([]),
    )
    rss: float = field(init=False, default=0.0)
    tss: float = field(init=False, default=0.0)
    ess: float = field(init=False, default=0.0)
    degrees_of_freedom: int = field(init=False, default=0)
    s2: float = field(init=False, default=0.0)
    r2: float = field(init=False, default=0.0)
    bic: float = field(init=False, default=0.0)
    param_covar: NDArray = field(
        init=False,
        default_factory=lambda: np.array([]),
    )

    def __post_init__(self) -> None:
        n_samples, d_features = self.X.shape[0], self.X.shape[1]
        ybar = self.y.mean()
        self.residuals = self.y - self.predictions
        self.rss = self.residuals @ self.residuals
        self.degrees_of_freedom = n_samples - d_features
        self.s2 = self.rss / self.degrees_of_freedom
        self.tss = (self.y - ybar) @ (self.y - ybar)
        self.ess = self.tss - self.rss
        self.r2 = self.ess / self.tss
        self.bic = n_samples * log(self.rss / n_samples) + d_features * log(
            n_samples
        )  # noqa
        self.param_covar = self._param_covar(self.X)

    def _param_covar(self, X: NDArray) -> NDArray:
        return np.linalg.inv(X.T @ X) * self.s2


@dataclass
class LinearRegression(LinearBase):
    """
    Implements the classic Linear Regression via ols
    Args:
    bias: indicates if intercept is added or not
    degree: degree of the polynomial, default=1
    regularization: supports 'l2' for ridge or 'l1' for lasso

    Attributes:
    theta:          Coefficient Weights after fitting
    residuals:      Number of Incorrect Predictions
    rss:            Residual sum of squares given by e'e
    tss:            Total sum of squares
    ess:            explained sum of squares
    r2:             Rsquared or proportion of variance
    s2:             Residual Standard error or RSE


    Notes:
    Class uses multiple estimation methods to estimate the ordinary
    lease squares problem min ||Ax - b||, where x = px1 is the parameter
    to be estimated, A=nxp matrix and b = nx1 vector is given
    - A naive implementation of (A'A)^-1 A'b = x is given
      but computing an inverse is expensive
    - A implementation based on QR decomposition is given based on
        min||Ax-b|| = min||Q'(QRx - b)|| = min||(Rx - Q'b)||
        based on decomposing nxp matrix A = QR, Q is orthogonal, R is upper
        triangular and ||Qv|| = (Qv)'Qv = v'Q'Qv = v'v = ||v||^2
    - A cholesky implementation is also included based on converting an n x p
        into a pxp matrix: A'A = A'b, then letting M = A'A & y = A'b, then
        solve Mx = y.  Leting M = U'U, we solve this by forward/backward sub
    """

    bias: bool = True
    degree: int = 1
    run: bool = field(init=False, default=False)
    theta: np.ndarray[np.float_] = field(
        init=False, default_factory=lambda: np.array([])
    )
    diagnostics: Optional[bool] = field(init=False, default=False)

    def _normal(self, A: NDArray, b: NDArray) -> NDArray:
        """Estimates parameters of regression model via normal method
        given by (A'A)^-1 A'b = x

        Parameters
        ----------
        A : NDArray
            design matrix
        b : NDarray, shape = (n_samples,)
            response variable

        Returns
        -------
        NDarray
            weights corresponding to linear transformation
        """
        return np.linalg.inv(A.T @ A) @ A.T @ b

    def _ols_qr(self, A: NDArray, b: NDArray) -> NDArray:
        """Estimates ||Ax - b|| via QR decomposition

        Parameters
        ----------
        A : NDArray
            [description]
        b : NDArray
            [description]

        Returns
        -------
        NDArray
            [description]
        """
        # min||(Rx - Q'b)
        q, r = np.linalg.qr(A)

        # solves by forward substitution
        return solve_triangular(r, q.T @ b)

    def _ols_cholesky(self, A: NDArray, b: NDArray) -> NDArray:
        """Estimates ||Ax - b|| via cholesky decomposition

        Parameters
        ----------
        A : ArrayLike
            feature matrix
        b : ArrayLike
            response variable

        Returns
        -------
        ArrayLike
            estimated weights
        """
        M = np.linalg.cholesky(A.T @ A)
        y = solve_triangular(M, A.T @ b, lower=True)
        return solve_triangular(M.T, y)

    def _get_solver(
        self,
        method: str = "ols-cholesky",
    ) -> Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        """Factory pattern that returns a solver

        Parameters
        ----------
        method : str, optional, default="ols-cholesky"
            fitting method, supports one of 'normal', 'ols-qr', 'ols-cholesky'

        Returns
        -------
        Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
            function that numerically solves Ax=b by minimizing
            ||Ax - b|| and returns weights x from fitting

        Raises
        ------
        TypeError
            if incorrect type is supplied
        """
        if method == "normal":
            return self._normal
        elif method == "ols-qr":
            return self._ols_qr
        elif method == "ols-cholesky":
            return self._ols_cholesky
        else:
            raise TypeError("method not available")

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        method: str = "normal",
        run_diagnostics: bool = False,
    ) -> LinearRegression:
        """Fits data via ordinary least squares

        Parameters
        ----------
        X : np.ndarray
            feature matrix
        y : np.ndarray
            response variable
        method : str, optional
            [description], by default "ols"

        Returns
        -------
        LinearRegression
            [description]
        """
        degree, bias = self.degree, self.bias
        X = self.make_polynomial(X, degree, bias)
        solver = self._get_solver(method)
        weights = solver(X, y)
        if weights is not None:
            self.theta = weights
            self.run = True

        if run_diagnostics:
            self.diagnostics = RegressionDiagnostics(
                X,
                y,
                weights,
                self.predict(X),
            )

        return self

    def predict(self, X: NDArray, theta: Optional[NDArray] = None) -> NDArray:
        """makes predictions of response variable given input params
        Args:
        X:
            shape = (n_samples, p_features)
            n_samples is number of instances
            p_features is number of features
            - if bias is true, a ones column is needed
        thetas:
            if initialized to None:
                uses estimated theta from fitting process
            if array is given:
                makes prediction from given thetas

        Returns:
        predicted values:
            shape = (n_samples,)
        """

        if theta is None and self.run:
            return X @ self.theta
        return X @ theta


@dataclass
class LinearRegressionMLE(LinearBase):
    """
    Implements linear regression via Maximum Likelihood Estimate
    Args:
    bias: indicates if intercept is added or not

    Attributes:
    theta:           Coefficient Weights after fitting
    residuals:       Number of Incorrect Predictions

    Notes:
    Class uses multiple estimation methods to estimate the oridiinary
    lease squares problem min ||Ax - b||, where x = px1, A=nxp, b = nx1
    - A implementation of MLE based on BFGS algorithm is given.  We are
        maximizing log(L(theta)):= L = -n/2 log(2pi *
        residual_std_error**2) - 0.5 ||Ax-b||
        This is same as minimizing 0.5||Ax-b||, the cost function J.
        The jacobian for regression is given by A'(Ax - b) -> (px1) vector
    - A implementation of MLE based on Newton-CG is provided.  The Hessian is:
        A'(Ax - b)A -> pxp matrix
    Todo
    - Levenberg-Marquardt Algorithm

    """

    bias: bool = True
    degree: int = 1
    run: bool = field(init=False, default=False)
    theta: Optional[NDArray] = field(init=False, default_factory=np.array([]))
    diagnostics: Optional[RegressionDiagnostics] = field(init=False)

    def _loglikelihood(self, true: ArrayLike, guess: ArrayLike) -> float:
        error = true - guess
        return 0.5 * (error**2).sum()

    def _objective_func(self, guess: NDArray, A: NDArray, b: NDArray) -> float:
        """the objective function to be minimized, returns estimated x for Ax=b
        Args:
        guess:
            initial guess for paramter x
            shape = {1, p_features}
            p_features is the number of columns of design matrix A

        A:
            the coefficient matrix
            shape = {n_samples, n_features}

        b:
            the response variable
            shape = {n_samples, 1}

        Returns:
        Scaler value from loglikelihood function
        """
        y_guess = self.predict(A, thetas=guess)
        f = self._loglikelihood(true=b, guess=y_guess)
        return f

    def _jacobian(self, guess: NDArray, A: NDArray, b: NDArray) -> NDArray:
        return A.T @ (guess @ A.T - b)

    def _hessian(self, guess: NDArray, A: NDArray, b: NDArray) -> ArrayLike:
        return A.T @ (A @ guess[:, np.newaxis] - b) @ A

    def _levenberg_marqdt(self) -> None:
        raise NotImplementedError()

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        method: str = "BFGS",
        run_diagnostics: bool = False,
    ) -> LinearRegressionMLE:
        """Fits training data via Maximum Likelihood Estimate

        Parameters
        ----------
        X : NDArray, shape=(n_samples, p_features)
            the design matrix
            n_samples is number of instances i.e rows
            p_features is number of features i.e columns
        y : NDArray, shape=(n_samples,)
            Target values
        method : str, optional, default='BFGS'
            fitting procedure
            Also supports 'Newton-CG'
        run_diagnostics : bool, optional, default=False
            whether to get the regression diagnostics

        Returns
        -------
        LinearRegressionMLE
            object after fitting
        """
        degree, bias = self.degree, self.bias
        X = self.make_polynomial(X, degree, bias)
        # generate random guess
        rng = np.random.RandomState(1)
        guess_params = rng.uniform(low=0, high=10, size=X.shape[1])
        if method == "BFGS":
            # doesn't require hessian
            self.theta = minimize(
                self._objective_func,
                guess_params,
                jac=self._jacobian,
                method="BFGS",
                options={"disp": True},
                args=(X, y),
            )
        elif method == "mle_newton_cg":
            # hess is optional but speeds up the iterations
            self.theta = minimize(
                self._objective_func,
                guess_params,
                jac=self._jacobian,
                hess=self._hessian,
                method="Newton-CG",
                options={"disp": True},
                args=(X, y),
            )
        self.run = True
        if run_diagnostics:
            self.diagnostics = compute_regression_diagnostics(
                X, y, self.theta, self.predict(X)
            )
        return self

    def predict(
        self,
        X: ArrayLike,
        theta: Optional[ArrayLike] = None,
    ) -> NDArray:
        """makes predictions of response variable given input params
        Args:
        X:
            shape = (n_samples, d_features)
            n_samples is number of instances
            d_features is number of features
            - if bias is true, a ones column is needed
        theta:
            if initialized to None:
                uses estimated theta from fitting process
            if array is given:
                it serves as initial guess for optimization

        Returns:
        predicted values:
            shape = (n_samples, 1)
        """
        if theta is None and self.run:
            return X @ self.theta
        return X @ theta


@dataclass
class LinearRegressionGD(LinearBase):
    """Implements the ols regression via Gradient Descent

    Args:
    eta:             Learning rate (between 0.0 and 1.0)
    n_iter:          passees over the training set
    random_state:    Random Number Generator seed
                     for random weight initilization

    Attributes:
    theta:           Weights after fitting
    residuals:       Number of incorrect predictions
    """

    eta: float = 0.001
    n_iter: int = 20
    random_state: int = 1
    bias: bool = True
    degree: int = 1
    cost: List[float] = field(init=False)
    theta: Optional[ArrayLike] = field(init=False)
    diagnostics: Optional[RegressionDiagnostics] = field(
        init=False,
        default=False,
    )
    run: bool = field(init=False, default=False)

    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearRegressionGD:
        """Fits model to training data via Gradient Descent

        Parameters
        ----------
        X : ArrayLike
            [description]
        y : ArrayLike
            [description]

        Returns
        -------
        LinearRegressionGD
            [description]
        """
        n_samples, p_features = X.shape[0], X.shape[1]
        self.theta = np.zeros(shape=1 + p_features)
        self.cost = []
        degree, bias = self.degree, self.bias
        X = self.make_polynomial(X, degree, bias)

        # todo - iterate until convergence
        for _ in range(self.n_iter):
            # calculate the error
            error = y - self.predict(X)
            self.theta += self.eta * X.T @ error / n_samples
            self.cost.append((error.T @ error) / (2.0 * n_samples))
        self.run = True
        return self

    def predict(
        self,
        X: ArrayLike,
        theta: Optional[ArrayLike] = None,
    ) -> NDArray:
        """Makes predictions of target variable given data

        Parameters
        ----------
        X : ArrayLike, shape=(n_samples, p_features)
            [description]
        theta : Optional[ArrayLike], optional, default=None
            weights of parameters in model

        Returns
        -------
        NDArray
            predictions of response given thetas
        """
        if theta is None:
            return X @ self.theta
        return X @ theta


def compute_regression_diagnostics(
    X: NDArray, y: NDArray, theta: NDArray, predictions: NDArray
) -> RegressionDiagnostics:
    return RegressionDiagnostics(X, y, theta, predictions)
