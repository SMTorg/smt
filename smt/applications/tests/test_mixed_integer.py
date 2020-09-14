import unittest
import numpy as np
from smt.applications.mixed_integer import (
    MixedIntegerContext,
    FLOAT,
    ENUM,
    INT,
    check_xspec_consistency,
)
from smt.problems import Sphere
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG


class TestMixedInteger(unittest.TestCase):
    def test_check_xspec_consistency(self):
        xtypes = [FLOAT, (ENUM, 3), INT]
        xlimits = [[-10, 10], ["blue", "red", "green"]]  # Bad dimension
        with self.assertRaises(ValueError):
            check_xspec_consistency(xtypes, xlimits)

        xtypes = [FLOAT, (ENUM, 3), INT]
        xlimits = [[-10, 10], ["blue", "red"], [-10, 10]]  # Bad enum
        with self.assertRaises(ValueError):
            check_xspec_consistency(xtypes, xlimits)

    def test_krg_mixed_5D(self):
        xtypes = [FLOAT, (ENUM, 3), INT]
        xlimits = [[-10, 10], ["blue", "red", "green"], [-10, 10]]
        mixint = MixedIntegerContext(xtypes, xlimits)

        sm = mixint.build_surrogate(KRG(print_prediction=False))
        sampling = mixint.build_sampling_method(LHS, criterion="m")

        fun = Sphere(ndim=5)
        xt = sampling(20)
        yt = fun(xt)
        sm.set_training_values(xt, yt)
        sm.train()

        x_out = mixint.fold_with_enum_indexes(xt)
        eq_check = True
        for i in range(x_out.shape[0]):
            if abs(float(x_out[i, :][2]) - int(float(x_out[i, :][2]))) > 10e-8:
                eq_check = False
            if not (x_out[i, :][1] == 0 or x_out[i, :][1] == 1 or x_out[i, :][1] == 2):
                eq_check = False
        self.assertTrue(eq_check)
