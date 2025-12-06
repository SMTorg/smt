import numpy as np
import unittest
import jenn
import smt


class TestGENN(unittest.TestCase):
    def test_rosenbrock(self):
        """Check GENN predictions on Rosenbrock function."""

        # Generate synthetic training data outside of SMT (using JENN)
        x_train, y_train, dydx_train = jenn.utilities.sample(
            f=jenn.synthetic_data.rosenbrock.compute,
            f_prime=jenn.synthetic_data.rosenbrock.compute_partials,
            m_random=0,
            m_levels=3,
            lb=[-np.pi, -np.pi],
            ub=[np.pi, np.pi],
        )

        # Generate synthetic test data outside of SMT (using JENN)
        x_test, y_test, dydx_test = jenn.utilities.sample(
            f=jenn.synthetic_data.rosenbrock.compute,
            f_prime=jenn.synthetic_data.rosenbrock.compute_partials,
            m_random=0,
            m_levels=100,
            lb=[-np.pi, -np.pi],
            ub=[np.pi, np.pi],
        )

        # SMT and JENN data structures are transposed from each other
        x_train = x_train.T
        y_train = y_train.T
        dydx_train = dydx_train.squeeze().T

        x_test = x_test.T
        y_test = y_test.T
        dydx_test = dydx_test.squeeze().T

        # Training model using SMT API as usual
        genn = smt.surrogate_models.GENN()
        genn.options["hidden_layer_sizes"] = [12, 12]
        genn.options["alpha"] = 0.01
        genn.options["lambd"] = 0.01
        genn.options["gamma"] = 1
        genn.options["num_iterations"] = 5000
        genn.options["is_backtracking"] = True
        genn.options["is_normalize"] = True
        genn.options["seed"] = 123
        genn.load_data(x_train, y_train, dydx_train)
        genn.train()

        # Predict test data
        y_pred = genn.predict_values(x_test)

        # Make sure the prediction is good
        rsquare = jenn.metrics.rsquare(y_pred.ravel(), y_test.ravel())
        tol = 0.99
        self.assertGreater(rsquare, tol, msg=f"R^2 = {rsquare:.3f} is less than {tol}")
