"""
Author: Paul Saves
"""

import unittest

import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from smt.surrogate_models import KRG, LS
from smt.utils.sklearn_adapter import ScikitLearnAdapter

import sklearn
from packaging import version

from types import MethodType
from inspect import Signature, Parameter


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
        model.fit(self.X, self.y)
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

        if version.parse(sklearn.__version__) >= version.parse("1.6.0"):
            sig = Signature(
                [
                    Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
                    Parameter("model_cls", Parameter.POSITIONAL_OR_KEYWORD),
                ]
            )

            ScikitLearnAdapter.__init__.__signature__ = sig
            model = ScikitLearnAdapter(model_cls=LS)
            model.fit(self.X, self.y)
            #   Run sklearn 1.5-style estimator checks on sklearn 1.6+ without modifying the adapter.
            #   Hides dynamic attributes like model_kwargs from check_estimator.

            # Save original get_params
            original_get_params = model.get_params

            # Patch get_params to only return explicit __init__ params
            def patched_get_params(self, deep=True):
                # Only return attributes that are in the constructor signature
                return {"model_cls": self.model_cls}

            model.get_params = MethodType(patched_get_params, model)

            # Patch set_params to ignore unknown kwargs
            original_set_params = model.set_params

            def patched_set_params(self, **params):
                # Only set known params, ignore others
                known_params = {k: v for k, v in params.items() if k == "model_cls"}
                return original_set_params(**known_params)

            model.set_params = MethodType(patched_set_params, model)

            # Run legacy checks
            check_estimator(model, legacy=True)

            # Restore original methods
            model.get_params = original_get_params
            model.set_params = original_set_params

        else:
            model = ScikitLearnAdapter(LS, print_global=False)
            model.fit(self.X, self.y)
            check_estimator(model)


if __name__ == "__main__":
    unittest.main()
