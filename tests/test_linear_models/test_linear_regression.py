# To run the test, go to root folder and type:  pytest tests --cov=neptunelearn/linear_models --cov-report=html
# another option is to use python -m coverage run -m pytest tests
import unittest

import numpy as np
from numpy.testing import assert_allclose
from sklearn import datasets

from neptunelearn.linear_models.linear_regression import (
    LinearRegression,
    LinearRegressionMLE,
    RegressionDiagnostics,
)


class TestLinearRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features, cls.target, cls.coef = datasets.make_regression(
            n_samples=10,
            n_features=4,
            n_informative=4,
            n_targets=1,
            coef=True,
            random_state=42,
        )
        cls.lr = LinearRegression(bias=False)

    def tearDown(self):
        self.lr.theta = np.array([])

    def test_linear_regression_normal(self):
        self.lr.fit(self.features, self.target, method="normal")
        assert_allclose(self.lr.theta, self.coef)

    def test_get_regression_diagnostics(self):
        self.lr.fit(
            self.features,
            self.target,
            method="normal",
            run_diagnostics=True,
        )
        assert_allclose(self.lr.theta, self.coef)
        self.assertIsNotNone(self.lr.diagnostics)
        self.assertIsInstance(self.lr.diagnostics, RegressionDiagnostics)

    def test_linear_regression_cholesky(self):
        self.lr.fit(self.features, self.target, method="ols-cholesky")
        assert_allclose(self.lr.theta, self.coef)

    def test_linear_regression_qr(self):
        self.lr.fit(self.features, self.target, method="ols-qr")
        assert_allclose(self.lr.theta, self.coef)

    def test_linear_regression_raises_invalid_method(self):
        with self.assertRaises(TypeError) as context:
            self.lr.fit(self.features, self.target, method="test")
        self.assertEqual(str(context.exception), "method not available")


class TestLinearRegressionMLE(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features, cls.target, cls.coef = datasets.make_regression(
            n_samples=10,
            n_features=4,
            n_informative=4,
            n_targets=1,
            coef=True,
            random_state=42,
        )
        cls.lr = LinearRegressionMLE(bias=False)


if __name__ == "__main__":
    unittest.main()
