"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest
import numpy as np
from smt.surrogate_models.krg_based import KrgBased

from smt.surrogate_models import KRG
import openmdao.api as om


# defining the toy example
def target_fun(x):
    return np.cos(5 * x)


class TestKrgBased(unittest.TestCase):
    def test_theta0_default_init(self):
        krg = KrgBased()
        krg.set_training_values(np.array([[1, 2, 3]]), np.array([[1]]))
        krg._check_param()
        self.assertTrue(np.array_equal(krg.options["theta0"], [1e-2, 1e-2, 1e-2]))

    def test_theta0_one_dim_init(self):
        krg = KrgBased(theta0=[2e-2])
        krg.set_training_values(np.array([[1, 2, 3]]), np.array([[1]]))
        krg._check_param()
        self.assertTrue(np.array_equal(krg.options["theta0"], [2e-2, 2e-2, 2e-2]))

    def test_theta0_erroneous_init(self):
        krg = KrgBased(theta0=[2e-2, 1e-2])
        krg.set_training_values(np.array([[1, 2]]), np.array([[1]]))  # correct
        krg._check_param()
        krg.set_training_values(np.array([[1, 2, 3]]), np.array([[1]]))  # erroneous
        self.assertRaises(ValueError, krg._check_param)

    def test_almost_squar_exp(self):
        nobs = 50  # number of obsertvations
        np.random.seed(0)  # a seed for reproducibility
        xt = np.random.uniform(size=nobs)  # design points

        # adding a random noise to observations
        yt = target_fun(xt) + np.random.normal(scale=0.05, size=nobs)

        # training the model with the option eval_noise= True
        sm = KRG(eval_noise=False, corr="pow_exp", pow_exp_power=1.9999)
        sm.set_training_values(xt, yt)

        self.assertWarns(UserWarning, sm.train)

    def test_less_almost_squar_exp(self):
        nobs = 50  # number of obsertvations
        np.random.seed(0)  # a seed for reproducibility
        xt = np.random.uniform(size=nobs)  # design points

        # adding a random noise to observations
        yt = target_fun(xt) + np.random.normal(scale=0.05, size=nobs)

        # training the model with the option eval_noise= True
        sm = KRG(eval_noise=False, corr="pow_exp", pow_exp_power=1.999)
        sm.set_training_values(xt, yt)
        sm.train()

        # predictions
        x = np.linspace(0, 1, 500).reshape(-1, 1)
        sm.predict_values(x)  # predictive mean
        sm.predict_variances(x)  # predictive variance
        sm.predict_derivatives(x, 0)  # predictive variance

        class test_krg_derivs(om.ExplicitComponent):
            def setup(self):
                # Inputs
                self.add_input("x", 0.0)
                # Outputs
                self.add_output("y", 0.0)

                # Other dependencies
                self.declare_partials(of="y", wrt=["x"])

            def compute(self, inputs, outputs):
                """Considering the entire rotor as a single disc that extracts
                velocity uniformly from the incoming flow and converts it to
                power."""

                x = inputs["x"]
                outputs["y"] = sm.predict_values(x)

            def compute_partials(self, inputs, partials):
                x = inputs["x"]
                partials["y", "x"] = sm.predict_derivatives(x, 0)

        p = om.Problem(model=om.Group())
        ivc = p.model.add_subsystem("vars", om.IndepVarComp())
        ivc.add_output("x", shape=1)
        p.model.add_subsystem("test_krg_derivs", test_krg_derivs())
        p.model.connect("vars.x", "test_krg_derivs.x")
        p.setup(force_alloc_complex=True)
        p.set_val("vars.x", val=1.0)
        p.run_model()
        cpd = p.check_partials(method="fd", compact_print=True, step=1.0e-7)
        derivs = list((list((list(cpd.values())[0]).values())[0]).values())
        self.assertTrue(abs(derivs[0] - derivs[1]) > 1000)

        # training the model with the option eval_noise= True
        sm = KRG(eval_noise=False, corr="pow_exp", pow_exp_power=1.99)
        sm.set_training_values(xt, yt)
        sm.train()

        # predictions
        x = np.linspace(0, 1, 500).reshape(-1, 1)
        sm.predict_values(x)  # predictive mean
        sm.predict_variances(x)  # predictive variance
        sm.predict_derivatives(x, 0)  # predictive variance

        p = om.Problem(model=om.Group())
        ivc = p.model.add_subsystem("vars", om.IndepVarComp())
        ivc.add_output("x", shape=1)
        p.model.add_subsystem("test_krg_derivs", test_krg_derivs())
        p.model.connect("vars.x", "test_krg_derivs.x")
        p.setup(force_alloc_complex=True)
        p.set_val("vars.x", val=1.0)
        p.run_model()
        cpd = p.check_partials(method="fd", compact_print=True, step=1.0e-7)
        derivs = list((list((list(cpd.values())[0]).values())[0]).values())
        self.assertTrue(abs(derivs[0] - derivs[1]) < 5e-2)


if __name__ == "__main__":
    unittest.main()
