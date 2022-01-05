# -*- coding: utf-8 -*-
"""
Created on Mon May 07 14:20:11 2018

@author: N. Bartoli
to consider only 1 fidelity level
"""

import matplotlib

matplotlib.use("Agg")

import unittest
import numpy as np
import unittest
import inspect

from collections import OrderedDict

from smt.problems import Sphere, TensorProduct
from smt.sampling_methods import LHS, FullFactorial

from smt.utils.sm_test_case import SMTestCase
from smt.utils.silence import Silence
from smt.utils import compute_rms_error
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS, GENN
from smt.applications.mfk import MFK, NestedLHS
from copy import deepcopy

print_output = False


class TestMFK(SMTestCase):


    @staticmethod
    def run_mfk_example_1fidelity():
        import numpy as np
        import matplotlib.pyplot as plt
        from smt.applications.mfk import MFK, NestedLHS

        #to consider only 1 fidelity level
        # high fidelity model
        def hf_function(x):
            import numpy as np

            return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)

        # Problem set up
        xlimits = np.array([[0.0, 1.0]])
        xdoes = NestedLHS(nlevel=1, xlimits=xlimits, random_state=0)
        xt_e = xdoes(7)[0]

        # Evaluate the HF function
        yt_e = hf_function(xt_e)
        

        sm = MFK(theta0=xt_e.shape[1] * [1.0])


        # high-fidelity dataset without name
        sm.set_training_values(xt_e, yt_e)

        # train the model
        sm.train()

        x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

        # query the outputs
        y = sm.predict_values(x)
        mse = sm.predict_variances(x)
        derivs = sm.predict_derivatives(x, kx=0)

        plt.figure()

        plt.plot(x, hf_function(x), label="reference")
        plt.plot(x, y, linestyle="-.", label="mean_gp")
        plt.scatter(xt_e, yt_e, marker="o", color="k", label="HF doe")
    

        plt.legend(loc=0)
        plt.ylim(-10, 17)
        plt.xlim(-0.1, 1.1)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        plt.show()


if __name__ == "__main__":
    unittest.main()
