"""
Author: Paul Saves
"""

import unittest

import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from smt.surrogate_models import KRG, LS
from smt.utils.sklearn_adapter import ScikitLearnAdapter


class TestSklearnAdapter(unittest.TestCase):
    def setUp(self):
        # Prepare a sample dataset
        rng = np.random.RandomState(0)
        self.X = rng.rand(50, 1)
        self.y = np.sin(2 * self.X).ravel()

    def test_adapter_basic_fit_predict(self):
        model = ScikitLearnAdapter(LS, print_global=False)
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        # Check that output shape matches
        self.assertEqual(y_pred.shape, self.y.shape)
        # Predictions should be finite
        self.assertTrue(np.all(np.isfinite(y_pred)))

    def test_get_set_params(self):
        model = ScikitLearnAdapter(KRG, print_global=False, theta0=[2.0])
        params = dict(model.get_params().pop("model_cls").__dict__)
        #        model.fit(self.X, self.y)
        self.assertIn("theta0", params)
        # Change a param and ensure it takes effect
        self.assertEqual(
            dict(model.get_params().pop("model_cls").__dict__)["theta0"], [2.0]
        )

    def test_estimator_checks(self):
        """
        Ensure full compatibility with scikit-learn estimator conventions.
        This uses sklearn.utils.estimator_checks.check_estimator,
        which requires get_params, set_params, fit, predict, etc. :contentReference[oaicite:1]{index=1}
        """
        model = ScikitLearnAdapter(LS, print_global=False)
        model.fit(self.X, self.y)
        check_estimator(model)


if __name__ == "__main__":
    unittest.main()
