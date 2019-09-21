"""
Author: Mohamed Amine Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
"""

import unittest
import matplotlib

matplotlib.use("Agg")

import numpy as np
from scipy import linalg
from smt.utils.sm_test_case import SMTestCase
from smt.utils import compute_rms_error

from smt.problems import WaterFlowLFidelity, WaterFlow
from smt.sampling_methods import LHS
from smt.applications import VFM
from smt.utils.misc import compute_rms_error


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
        optionsB = {"theta0": [1e-2] * ndim, "print_prediction": False, "deriv": False}

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
        rms_error = compute_rms_error(vfm, xtest, ytest)
        self.assert_error(rms_error, 0.0, 3e-1)

    @staticmethod
    def run_vfm_example(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import linalg
        from smt.utils import compute_rms_error

        from smt.problems import WaterFlowLFidelity, WaterFlow
        from smt.sampling_methods import LHS
        from smt.applications import VFM

        # Problem set up
        ndim = 8
        ntest = 500
        ncomp = 1
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
        optionsB = {"theta0": [1e-2] * ndim, "print_prediction": False, "deriv": False}

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


if __name__ == "__main__":
    unittest.main()
