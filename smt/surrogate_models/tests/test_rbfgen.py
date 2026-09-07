import numpy as np
import unittest

from smt.surrogate_models.rbfgen import RBFGEN_AVAILABLE
from smt.surrogate_models import RBFGen
from smt.utils.nn_lossterms import MonotonicityLossTerm, PositivityLossTerm


class TestRBFGen(unittest.TestCase):
    @unittest.skipIf(not RBFGEN_AVAILABLE, "RBFGen not available")
    def test_basic_predictions(self):
        # This test asserts that the R^2 accuracy
        # of RBFGen predictions on a test set is above 0.90.

        # Implementation:
        xt = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1)
        yt = np.sin(xt) + 1.5

        sm = RBFGen(
            epochs=200, learning_rate=5e-2, rbf_m_centers=20, print_global=False
        )
        sm.set_training_values(xt, yt)
        sm.add_loss_term(PositivityLossTerm(x_train=xt, loss_term_weight=0.0))
        sm.train()

        x_test = np.linspace(0.5, 2 * np.pi - 0.5, 50).reshape(-1, 1)
        y_test = np.sin(x_test) + 1.5

        y_pred = sm.predict_values(x_test)

        # Calculate R^2
        sse = np.sum((y_test - y_pred) ** 2)
        sst = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - sse / sst

        self.assertGreater(r2, 0.90, f"R^2 is too low: {r2}")

    @unittest.skipIf(not RBFGEN_AVAILABLE, "RBFGen not available")
    def test_variance_prediction(self):
        # This test ensures that `predict_variances` returns an array
        # of the correct shape and that all variance values are non-negative.

        # Implementation:
        xt = np.array([[0.0], [1.0], [2.0]])
        yt = np.array([[0.0], [1.0], [0.0]])

        sm = RBFGen(epochs=50, learning_rate=5e-2, rbf_m_centers=10, print_global=False)
        sm.set_training_values(xt, yt)
        sm.add_loss_term(PositivityLossTerm(x_train=xt, loss_term_weight=0.0))
        sm.train()

        x_test = np.array([[0.5], [1.5]])
        s2 = sm.predict_variances(x_test)

        self.assertEqual(s2.shape, (2, 1))
        self.assertTrue(np.all(s2 >= 0.0), "Variances should be non-negative")

    @unittest.skipIf(not RBFGEN_AVAILABLE, "RBFGen not available")
    def test_loss_terms(self):
        # This test adds a MonotonicityLossTerm and a PositivityLossTerm to the model,
        # trains it, and asserts that no errors are thrown during the training process,
        # and that basic predictions can still be made.

        # Implementation:
        xt = np.array([[0.0], [2.0], [4.0]])
        yt = np.array([[0.5], [2.0], [3.5]])

        sm = RBFGen(epochs=50, learning_rate=5e-2, rbf_m_centers=10, print_global=False)
        sm.set_training_values(xt, yt)

        sm.add_loss_term(MonotonicityLossTerm(x_train=xt, random_base_points=True))
        sm.add_loss_term(PositivityLossTerm(x_train=xt))

        sm.train()

        x_test = np.array([[1.0], [3.0]])
        y_pred = sm.predict_values(x_test)

        self.assertEqual(y_pred.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
