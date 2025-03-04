"""
Author: Mohamed Amine Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

import unittest

try:
    import matplotlib

    matplotlib.use("Agg")
    NO_MATPLOTLIB = False
except ImportError:
    NO_MATPLOTLIB = True

import numpy as np

from smt.applications import VFM
from smt.examples.rans_crm_wing.rans_crm_wing import (
    get_rans_crm_wing,
)
from smt.problems import WaterFlow, WaterFlowLFidelity
from smt.sampling_methods import LHS
from smt.utils.misc import compute_relative_error
from smt.utils.silence import Silence
from smt.utils.sm_test_case import SMTestCase


def setupCRM(LF_candidate="QP", Bridge_candidate="KRG", type_bridge="Additive"):
    xt, yt, _ = get_rans_crm_wing()

    optionsLF = {}
    optionsB = {}

    M = VFM(
        type_bridge=type_bridge,
        name_model_LF=LF_candidate,
        name_model_bridge=Bridge_candidate,
        X_LF=xt,
        y_LF=yt,
        X_HF=xt,
        y_HF=yt,
        options_LF=optionsLF,
        options_bridge=optionsB,
    )

    return M, xt


class TestVFM(SMTestCase):
    def test_vfm(self):
        # Problem set up
        ndim = 8
        ntest = 500
        ndoeLF = int(10 * ndim)
        ndoeHF = int(3)
        funLF = WaterFlowLFidelity(ndim=ndim)
        funHF = WaterFlow(ndim=ndim)
        deriv1 = True
        deriv2 = True
        LF_candidate = "QP"
        Bridge_candidate = "KRG"
        type_bridge = "Multiplicative"
        optionsLF = {}
        optionsB = {
            "theta0": [1e-2] * ndim,
            "print_prediction": False,
            "deriv": False,
            "hyper_opt": "Cobyla",
        }

        # Construct low/high fidelity data and validation points
        sampling = LHS(xlimits=funLF.xlimits, criterion="m", random_state=42)
        xLF = sampling(ndoeLF)
        yLF = funLF(xLF)
        if deriv1:
            dy_LF = np.zeros((ndoeLF, 1))
            for i in range(ndim):
                yd = funLF(xLF, kx=i)
                dy_LF = np.concatenate((dy_LF, yd), axis=1)

        sampling = LHS(xlimits=funHF.xlimits, criterion="m", random_state=43)
        xHF = sampling(ndoeHF)
        yHF = funHF(xHF)
        if deriv2:
            dy_HF = np.zeros((ndoeHF, 1))
            for i in range(ndim):
                yd = funHF(xHF, kx=i)
                dy_HF = np.concatenate((dy_HF, yd), axis=1)

        xtest = sampling(ntest)
        ytest = funHF(xtest)
        dytest = np.zeros((ntest, ndim))
        for i in range(ndim):
            dytest[:, i] = funHF(xtest, kx=i).T

        # Initialize VFM
        vfm = VFM(
            type_bridge=type_bridge,
            name_model_LF=LF_candidate,
            name_model_bridge=Bridge_candidate,
            X_LF=xLF,
            y_LF=yLF,
            X_HF=xHF,
            y_HF=yHF,
            options_LF=optionsLF,
            options_bridge=optionsB,
            dy_LF=dy_LF,
            dy_HF=dy_HF,
        )

        # Prediction of the validation points
        rms_error = compute_relative_error(vfm, xtest, ytest)
        self.assert_error(rms_error, 0.0, 3e-1)

    @staticmethod
    def run_vfm_example(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.applications import VFM
        from smt.problems import WaterFlow, WaterFlowLFidelity
        from smt.sampling_methods import LHS

        # Problem set up
        ndim = 8
        ntest = 500
        ndoeLF = int(10 * ndim)
        ndoeHF = int(3)
        funLF = WaterFlowLFidelity(ndim=ndim)
        funHF = WaterFlow(ndim=ndim)
        deriv1 = True
        deriv2 = True
        LF_candidate = "QP"
        Bridge_candidate = "KRG"
        type_bridge = "Multiplicative"
        optionsLF = {}
        optionsB = {
            "theta0": [1e-2] * ndim,
            "print_prediction": False,
            "deriv": False,
            "hyper_opt": "Cobyla",
        }

        # Construct low/high fidelity data and validation points
        sampling = LHS(xlimits=funLF.xlimits, criterion="m")
        xLF = sampling(ndoeLF)
        yLF = funLF(xLF)
        if deriv1:
            dy_LF = np.zeros((ndoeLF, 1))
            for i in range(ndim):
                yd = funLF(xLF, kx=i)
                dy_LF = np.concatenate((dy_LF, yd), axis=1)

        sampling = LHS(xlimits=funHF.xlimits, criterion="m")
        xHF = sampling(ndoeHF)
        yHF = funHF(xHF)
        if deriv2:
            dy_HF = np.zeros((ndoeHF, 1))
            for i in range(ndim):
                yd = funHF(xHF, kx=i)
                dy_HF = np.concatenate((dy_HF, yd), axis=1)

        xtest = sampling(ntest)
        ytest = funHF(xtest)
        dytest = np.zeros((ntest, ndim))
        for i in range(ndim):
            dytest[:, i] = funHF(xtest, kx=i).T

        # Initialize the extension VFM
        M = VFM(
            type_bridge=type_bridge,
            name_model_LF=LF_candidate,
            name_model_bridge=Bridge_candidate,
            X_LF=xLF,
            y_LF=yLF,
            X_HF=xHF,
            y_HF=yHF,
            options_LF=optionsLF,
            options_bridge=optionsB,
            dy_LF=dy_LF,
            dy_HF=dy_HF,
        )

        # Prediction of the validation points
        y = M.predict_values(x=xtest)

        plt.figure()
        plt.plot(ytest, ytest, "-.")
        plt.plot(ytest, y, ".")
        plt.xlabel(r"$y$ True")
        plt.ylabel(r"$y$ prediction")
        plt.show()

    def test_KRG_KRG_additive(self):
        with Silence():
            M, xt = setupCRM(
                LF_candidate="KRG", Bridge_candidate="KRG", type_bridge="Additive"
            )

        with Silence():
            yp = M.predict_values(np.atleast_2d(xt[0]))
            dyp = M.predict_derivatives(np.atleast_2d(xt[0]), kx=0)
        self.assert_error(yp, np.array([[0.015368, 0.367424]]), atol=2e-2, rtol=3e-2)
        self.assert_error(
            dyp,
            np.array([[0.31539861, 7.30043182]]),
            atol=5e-1,
            rtol=5e-2,
        )

    def test_QP_KRG_additive(self):
        with Silence():
            M, xt = setupCRM(
                LF_candidate="QP", Bridge_candidate="KRG", type_bridge="Additive"
            )

        with Silence():
            yp = M.predict_values(np.atleast_2d(xt[0]))
            dyp = M.predict_derivatives(np.atleast_2d(xt[0]), kx=0)

        self.assert_error(
            yp, np.array([[0.01535549, 0.36720433]]), atol=1e-2, rtol=1e-2
        )
        self.assert_error(
            dyp, np.array([[0.27030465, 5.89010571]]), atol=5e-1, rtol=5e-2
        )

    def test_KRG_KRG_mult(self):
        with Silence():
            M, xt = setupCRM(
                LF_candidate="KRG", Bridge_candidate="KRG", type_bridge="Multiplicative"
            )

        with Silence():
            yp = M.predict_values(np.atleast_2d(xt[0]))
            dyp = M.predict_derivatives(np.atleast_2d(xt[0]), kx=0)

        self.assert_error(
            yp, np.array([[0.01536264, 0.36713384]]), atol=2e-2, rtol=3e-2
        )
        self.assert_error(
            dyp, np.array([[0.29039224, 7.16030401]]), atol=5e-1, rtol=5e-2
        )

    def test_QP_KRG_mult(self):
        with Silence():
            M, xt = setupCRM(
                LF_candidate="QP", Bridge_candidate="KRG", type_bridge="Multiplicative"
            )

        with Silence():
            yp = M.predict_values(np.atleast_2d(xt[0]))
            dyp = M.predict_derivatives(np.atleast_2d(xt[0]), kx=0)

        self.assert_error(
            yp, np.array([[0.01536423, 0.36742269]]), atol=3e-1, rtol=1e-2
        )
        self.assert_error(
            dyp, np.array([[0.19562493, 5.39545518]]), atol=3e-1, rtol=1e-2
        )


if __name__ == "__main__":
    unittest.main()
