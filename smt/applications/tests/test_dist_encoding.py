import unittest
import numpy as np
from smt.surrogate_models import KRG
from smt.design_space import DesignSpace, CategoricalVariable, FloatVariable
from smt.surrogate_models.krg_based.kernel_types import MixIntKernelType
from smt.sampling_methods import LHS
from scipy.stats import spearmanr


class TestDistEncoding(unittest.TestCase):
    def test_dist_encoding_krg(self):
        """Basic 1D categorical check: preserve order and accuracy."""
        n_per_level = 5
        X = np.repeat([0, 1, 2], n_per_level).reshape(-1, 1)

        # y for level 0: ~1.0, level 1: ~5.0, level 2: ~10.0
        y = np.concatenate(
            [
                np.random.normal(1.0, 0.2, n_per_level),
                np.random.normal(5.0, 0.2, n_per_level),
                np.random.normal(10.0, 0.2, n_per_level),
            ]
        ).reshape(-1, 1)

        ds = DesignSpace([CategoricalVariable(values=["A", "B", "C"])])

        sm = KRG(
            design_space=ds,
            categorical_kernel=MixIntKernelType.DIST_ENCODING,
            print_global=False,
            n_start=5,
        )

        sm.set_training_values(X, y)
        sm.train()

        X_test = np.array([[0], [1], [2]])
        y_test = np.array([1.0, 5.0, 10.0])
        y_pred = sm.predict_values(X_test)

        corr, _ = spearmanr(y_test, y_pred.flatten())
        self.assertGreater(
            corr, 0.9, "Predictions should be highly correlated with levels"
        )
        np.testing.assert_allclose(y_pred.flatten(), y_test, atol=1.0)

    def test_dist_encoding_beta_effect(self):
        """Verify that beta scales the Wasserstein distance correctly (W2^beta)."""
        n_per_level = 5
        X = np.repeat([0, 1], n_per_level).reshape(-1, 1)
        y = np.concatenate(
            [
                np.random.normal(0.0, 0.1, n_per_level),
                np.random.normal(5.0, 0.1, n_per_level),
            ]
        ).reshape(-1, 1)

        ds = DesignSpace([CategoricalVariable(values=["A", "B"])])

        sm1 = KRG(
            design_space=ds,
            categorical_kernel=MixIntKernelType.DIST_ENCODING,
            categorical_kernel_beta=1.0,
            print_global=False,
            n_start=5,
        )
        sm1.set_training_values(X, y)
        sm1.train()

        sm2 = KRG(
            design_space=ds,
            categorical_kernel=MixIntKernelType.DIST_ENCODING,
            categorical_kernel_beta=2.0,
            print_global=False,
            n_start=5,
        )
        sm2.set_training_values(X, y)
        sm2.train()

        dist1 = sm1._mix_int_corr.de_encoder.get_w2_distance(0, 0, 1)
        dist2 = sm2._mix_int_corr.de_encoder.get_w2_distance(0, 0, 1)

        # dist2 should be dist1^2 (since beta2=2.0, beta1=1.0)
        self.assertAlmostEqual(dist2, dist1**2, places=5)

    def test_grouped_levels_da_veiga_reproduction(self):
        """
        Reproduction of the grouped levels benchmark from Da Veiga (2025).
        Ensures DE captures similarity between categorical levels within a group.
        """
        n_levels = 12
        n_per_level = 5

        def complex_fun(x):
            x_cat = x[:, 0].astype(int)
            x_cont = x[:, 1]
            y = np.zeros_like(x_cat, dtype=float)
            y[x_cat < 4] = 1.0 * x_cont[x_cat < 4]  # Group 1
            y[(x_cat >= 4) & (x_cat < 8)] = (
                1.1 * x_cont[(x_cat >= 4) & (x_cat < 8)]
            )  # Group 2
            y[x_cat >= 8] = 50.0 * x_cont[x_cat >= 8]  # Group 3 (Far)
            return y.reshape(-1, 1) + np.random.normal(0, 0.01, (len(x_cat), 1))

        ds = DesignSpace(
            [
                CategoricalVariable(values=[str(i) for i in range(n_levels)]),
                FloatVariable(0, 1),
            ]
        )

        lhs = LHS(xlimits=np.array([[0, 1]]), seed=42)
        x_cont_train = lhs(n_levels * n_per_level)
        x_cat_train = np.repeat(np.arange(n_levels), n_per_level).reshape(-1, 1)
        xtrain = np.hstack([x_cat_train, x_cont_train])
        ytrain = complex_fun(xtrain)

        # Fit DIST_ENCODING (DE)
        sm_de = KRG(
            design_space=ds,
            categorical_kernel=MixIntKernelType.DIST_ENCODING,
            print_global=False,
            n_start=5,
        )
        sm_de.set_training_values(xtrain, ytrain)
        sm_de.train()

        # Fit GOWER (Standard)
        sm_gower = KRG(
            design_space=ds,
            categorical_kernel=MixIntKernelType.GOWER,
            print_global=False,
            n_start=5,
        )
        sm_gower.set_training_values(xtrain, ytrain)
        sm_gower.train()

        # Evaluate
        x_cont_test = lhs(50)
        x_cat_test = np.random.choice(np.arange(n_levels), 50).reshape(-1, 1)
        xtest = np.hstack([x_cat_test, x_cont_test])
        ytest = complex_fun(xtest)

        mse_de = np.mean((ytest - sm_de.predict_values(xtest)) ** 2)
        mse_gower = np.mean((ytest - sm_gower.predict_values(xtest)) ** 2)

        self.assertLess(
            mse_de, mse_gower, "DE should outperform Gower on structured mixed problems"
        )

    def test_learning_curve(self):
        """Check that MSE decreases with more points for DIST_ENCODING."""
        ds = DesignSpace([CategoricalVariable(["A", "B", "C"])])

        def fun(x):
            x_cat = x[:, 0].astype(int)
            offsets = np.array([0.0, 5.0, 10.0])
            return offsets[x_cat].reshape(-1, 1)

        mses = []
        xtest = np.array([[0], [1], [2]])
        ytest = fun(xtest)

        for n_per_cat in [2, 5, 10]:
            xtrain = np.repeat([0, 1, 2], n_per_cat).reshape(-1, 1)
            ytrain = fun(xtrain) + np.random.normal(0, 0.01, (n_per_cat * 3, 1))

            sm = KRG(
                design_space=ds,
                categorical_kernel=MixIntKernelType.DIST_ENCODING,
                categorical_kernel_beta=1.0,
                print_global=False,
                n_start=1,
            )
            sm.set_training_values(xtrain, ytrain)
            sm.train()

            mse = np.mean((ytest - sm.predict_values(xtest)) ** 2)
            mses.append(mse)

        self.assertLess(mses[-1], 0.01, "Final MSE should be very small")


if __name__ == "__main__":
    unittest.main()
