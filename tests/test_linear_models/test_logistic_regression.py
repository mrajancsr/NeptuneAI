import unittest

import numpy as np

from neptunelearn.linear_models.logistic_regression import LogisticRegression


class TestLogisticRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lg = LogisticRegression()

    def tearDown(self):
        self.lg.theta = np.array([])

    def test_sigmoid(self):
        res = self.lg.sigmoid(0)
        self.assertEqual(res, 0.5)
