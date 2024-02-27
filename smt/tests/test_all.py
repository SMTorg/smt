"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
        Dr. Mohamed A. Bouhlel <mbouhlel@umich>

This package is distributed under New BSD license.
"""

import numpy as np
import unittest
import inspect

from collections import OrderedDict

from smt.problems import TensorProduct
from smt.sampling_methods import LHS

from smt.utils.sm_test_case import SMTestCase
from smt.utils.silence import Silence
from smt.utils.misc import compute_rms_error
from smt.surrogate_models import (
    LS,
    QP,
    KPLS,
    KRG,
    KPLSK,
    GEKPLS,
    GPX,
    GENN,
    DesignSpace,
)

try:
    from smt.surrogate_models import IDW, RBF, RMTC, RMTB

    COMPILED_AVAILABLE = True
except ImportError:
    COMPILED_AVAILABLE = False


print_output = False


class Test(SMTestCase):
    def setUp(self):
        ndim = 3
        nt = 100
        ne = 100
        ncomp = 1

        problems = OrderedDict()
        problems["exp"] = TensorProduct(ndim=ndim, func="exp")
        problems["tanh"] = TensorProduct(ndim=ndim, func="tanh")
        problems["cos"] = TensorProduct(ndim=ndim, func="cos")

        sms = OrderedDict()
        sms["LS"] = LS()
        sms["QP"] = QP()
        sms["GPX"] = GPX()
        sms["KRG"] = KRG(theta0=[1e-2] * ndim)
        sms["KPLS"] = KPLS(theta0=[1e-2] * ncomp, n_comp=ncomp)
        sms["KPLSK"] = KPLSK(theta0=[1] * ncomp, n_comp=ncomp)
        sms["MGP"] = KPLSK(theta0=[1e-2] * ncomp, n_comp=ncomp)
        sms["GEKPLS"] = GEKPLS(theta0=[1e-2] * 2, n_comp=2, delta_x=1e-1)
        sms["GENN"] = GENN(
            num_iterations=1000,
            hidden_layer_sizes=[
                24,
            ],
            alpha=1e-1,
            lambd=1e-2,
            is_backtracking=True,
            is_normalize=True,
        )

        if COMPILED_AVAILABLE:
            sms["IDW"] = IDW()
            sms["RBF"] = RBF()
            sms["RMTC"] = RMTC()
            sms["RMTB"] = RMTB()

        t_errors = {}
        t_errors["LS"] = 1.0
        t_errors["QP"] = 1.0
        t_errors["GPX"] = 1.2
        t_errors["KRG"] = 1.2
        t_errors["MFK"] = 1e0
        t_errors["KPLS"] = 1.2
        t_errors["KPLSK"] = 1e0
        t_errors["MGP"] = 1e0
        t_errors["GEKPLS"] = 1.4
        t_errors["GENN"] = 1.2
        if COMPILED_AVAILABLE:
            t_errors["IDW"] = 1e0
            t_errors["RBF"] = 1e-2
            t_errors["RMTC"] = 1e-1
            t_errors["RMTB"] = 1e-1

        e_errors = {}
        e_errors["LS"] = 1.5
        e_errors["QP"] = 1.5
        e_errors["GPX"] = 2e-2
        e_errors["KRG"] = 2e-2
        e_errors["MFK"] = 2e-2
        e_errors["KPLS"] = 2e-2
        e_errors["KPLSK"] = 2e-2
        e_errors["MGP"] = 2e-2
        e_errors["GEKPLS"] = 2e-2
        e_errors["GENN"] = 3e-2
        if COMPILED_AVAILABLE:
            e_errors["IDW"] = 1e0
            e_errors["RBF"] = 1e0
            e_errors["RMTC"] = 2e-1
            e_errors["RMTB"] = 3e-1

        self.nt = nt
        self.ne = ne
        self.ndim = ndim
        self.problems = problems
        self.sms = sms
        self.t_errors = t_errors
        self.e_errors = e_errors

    def run_test(self):
        method_name = inspect.stack()[1][3]
        pname = method_name.split("_")[1]
        sname = method_name.split("_")[2]

        prob = self.problems[pname]
        sampling = LHS(xlimits=prob.xlimits, random_state=42)

        xt = sampling(self.nt)
        yt = prob(xt)
        print(prob(xt, kx=0).shape)
        for i in range(self.ndim):
            yt = np.concatenate((yt, prob(xt, kx=i)), axis=1)

        xe = sampling(self.ne)
        ye = prob(xe)

        sm0 = self.sms[sname]

        sm = sm0.__class__()

        sm.options = sm0.options.clone()
        if sm.options.is_declared("design_space"):
            sm.options["design_space"] = DesignSpace(prob.xlimits)
        if sm.options.is_declared("xlimits"):
            sm.options["xlimits"] = prob.xlimits
        sm.options["print_global"] = False

        if sname in ["KPLS", "KRG", "KPLSK", "GEKPLS"]:
            optname = method_name.split("_")[3]
            sm.options["hyper_opt"] = optname

        sm.set_training_values(xt, yt[:, 0])
        if sm.supports["training_derivatives"]:
            for i in range(self.ndim):
                sm.set_training_derivatives(xt, yt[:, i + 1], i)

        with Silence():
            sm.train()

        t_error = compute_rms_error(sm)
        e_error = compute_rms_error(sm, xe, ye)

        if sm.supports["variances"]:
            sm.predict_variances(xe)

        # Some test case tolerance relaxations wrt to global tolerance values
        if pname == "cos":
            self.assertLessEqual(e_error, self.e_errors[sname] + 1.6)
        elif pname == "tanh" and sname in ["KPLS", "RMTB"]:
            self.assertLessEqual(e_error, self.e_errors[sname] + 0.4)
        elif pname == "exp" and sname in ["GENN"]:
            self.assertLessEqual(e_error, 1e-1)
        elif pname == "exp" and sname in ["RMTB"]:
            self.assertLessEqual(e_error, self.e_errors[sname] + 0.5)
        else:
            self.assertLessEqual(e_error, self.e_errors[sname])

        self.assertLessEqual(t_error, self.t_errors[sname])

    def test_exp_LS(self):
        self.run_test()

    def test_exp_QP(self):
        self.run_test()

    def test_exp_GPX(self):
        self.run_test()

    def test_exp_KRG_Cobyla(self):
        self.run_test()

    def test_exp_KRG_TNC(self):
        self.run_test()

    def test_exp_KPLS_Cobyla(self):
        self.run_test()

    def test_exp_KPLS_TNC(self):
        self.run_test()

    def test_exp_KPLSK_Cobyla(self):
        self.run_test()

    def test_exp_KPLSK_TNC(self):
        self.run_test()

    def test_exp_MGP(self):
        self.run_test()

    def test_exp_GEKPLS_Cobyla(self):
        self.run_test()

    def test_exp_GEKPLS_TNC(self):
        self.run_test()

    def test_exp_GENN(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_exp_IDW(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_exp_RBF(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_exp_RMTC(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_exp_RMTB(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: tanh

    def test_tanh_LS(self):
        self.run_test()

    def test_tanh_QP(self):
        self.run_test()

    def test_tanh_GPX(self):
        self.run_test()

    def test_tanh_KRG_Cobyla(self):
        self.run_test()

    def test_tanh_KRG_TNC(self):
        self.run_test()

    def test_tanh_KPLS_Cobyla(self):
        self.run_test()

    def test_tanh_KPLS_TNC(self):
        self.run_test()

    def test_tanh_KPLSK_Cobyla(self):
        self.run_test()

    def test_tanh_KPLSK_TNC(self):
        self.run_test()

    def test_tanh_MGP(self):
        self.run_test()

    def test_tanh_GEKPLS_Cobyla(self):
        self.run_test()

    def test_tanh_GEKPLS_TNC(self):
        self.run_test()

    def test_tanh_GENN(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_tanh_IDW(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_tanh_RBF(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_tanh_RMTC(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_tanh_RMTB(self):
        self.run_test()

    # --------------------------------------------------------------------
    # Function: cos

    def test_cos_LS(self):
        self.run_test()

    def test_cos_QP(self):
        self.run_test()

    def test_cos_GPX(self):
        self.run_test()

    def test_cos_KRG_Cobyla(self):
        self.run_test()

    def test_cos_KRG_TNC(self):
        self.run_test()

    def test_cos_KPLS_Cobyla(self):
        self.run_test()

    def test_cos_KPLS_TNC(self):
        self.run_test()

    def test_cos_KPLSK_Cobyla(self):
        self.run_test()

    def test_cos_KPLSK_TNC(self):
        self.run_test()

    def test_cos_MGP(self):
        self.run_test()

    def test_cos_GEKPLS_Cobyla(self):
        self.run_test()

    def test_cos_GEKPLS_TNC(self):
        self.run_test()

    def test_cos_GENN(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_cos_IDW(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_cos_RBF(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_cos_RMTC(self):
        self.run_test()

    @unittest.skipIf(not COMPILED_AVAILABLE, "Compiled Fortran libraries not available")
    def test_cos_RMTB(self):
        self.run_test()


if __name__ == "__main__":
    print_output = True
    print("%6s %8s %18s %18s" % ("SM", "Problem", "Train. pt. error", "Test pt. error"))
    unittest.main()
