import os
import unittest

import numpy as np

from smt.problems import Sphere
from smt.sampling_methods import LHS
from smt.surrogate_models import (
    GEKPLS,
    GENN,
    IDW,
    KPLS,
    KPLSK,
    KRG,
    LS,
    MGP,
    QP,
    RBF,
    RMTB,
    RMTC,
    SGP,
)


class TestSaveLoad(unittest.TestCase):
    def test_save_load_GEKPLS(self):
        filename = "sm_save_test"
        ndim = 2
        fun = Sphere(ndim=ndim)

        sampling = LHS(xlimits=fun.xlimits, criterion="m")
        xt = sampling(20)
        yt = fun(xt)

        for i in range(ndim):
            yd = fun(xt, kx=i)
            yt = np.concatenate((yt, yd), axis=1)

        X = np.arange(fun.xlimits[0, 0], fun.xlimits[0, 1], 0.25)
        Y = np.arange(fun.xlimits[1, 0], fun.xlimits[1, 1], 0.25)
        X, Y = np.meshgrid(X, Y)
        Z1 = np.zeros((X.shape[0], X.shape[1]))
        Z2 = np.zeros((X.shape[0], X.shape[1]))

        sm = GEKPLS(print_global=False)
        sm.set_training_values(xt, yt[:, 0])
        for i in range(ndim):
            sm.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)
        sm.train()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z1[i, j] = sm.predict_values(
                    np.hstack((X[i, j], Y[i, j])).reshape((1, 2))
                ).item()
        sm.save(filename)

        sm2 = GEKPLS.load(filename)
        self.assertIsNotNone(sm2)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z2[i, j] = sm2.predict_values(
                    np.hstack((X[i, j], Y[i, j])).reshape((1, 2))
                ).item()

        np.testing.assert_allclose(Z1, Z2)

        os.remove(filename)

    def test_save_load_surrogates(self):
        surrogates = [KRG, KPLS, KPLSK, MGP, SGP, QP, GENN, LS]
        rng = np.random.RandomState(1)
        N_inducing = 30
        num = 100

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])
        x = np.linspace(0.0, 4.0, num).reshape(-1, 1)

        for surrogate in surrogates:
            filename = "sm_save_test"

            sm = surrogate(print_global=False)
            sm.set_training_values(xt, yt)

            if surrogate == SGP:
                sm.Z = 2 * rng.rand(N_inducing, 1) - 1
                sm.set_inducing_inputs(Z=sm.Z)

            sm.train()
            y1 = sm.predict_values(x)
            sm.save(filename)

            sm2 = surrogate.load(filename)
            y2 = sm2.predict_values(x)

            np.testing.assert_allclose(y1, y2)

            os.remove(filename)

    def test_save_load_surrogates_cpp(self):
        surrogates_cpp = [RBF, RMTC, RMTB, IDW]

        num = 100
        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])
        x = np.linspace(0.0, 4.0, num)
        xlimits = np.array([[0.0, 4.0]])

        filename = "sm_save_test"

        for surrogate in surrogates_cpp:
            if surrogate == RMTB:
                sm = RMTB(
                    xlimits=xlimits,
                    order=4,
                    num_ctrl_pts=20,
                    energy_weight=1e-15,
                    regularization_weight=0.0,
                )
            elif surrogate == RMTC:
                sm = RMTC(
                    xlimits=xlimits,
                    num_elements=6,
                    energy_weight=1e-15,
                    regularization_weight=0.0,
                )
            elif surrogate == RBF:
                sm = RBF(d0=5)
            else:
                sm = IDW(p=2)

            sm.set_training_values(xt, yt)
            sm.train()

            y1 = sm.predict_values(x)
            sm.save(filename)

            sm2 = surrogate.load(filename)
            y2 = sm2.predict_values(x)

            np.testing.assert_allclose(y1, y2)

            os.remove(filename)


if __name__ == "__main__":
    unittest.main()
