

import numpy as np
import unittest

from smt.utils.sm_test_case import SMTestCase
from smt.utils.silence import Silence
from smt.surrogate_models import QP, KRG
from smt.examples.rans_crm_wing.rans_crm_wing import (
    get_rans_crm_wing,
    plot_rans_crm_wing,
)

def setup_KRG():

    xt, yt, xlimits = get_rans_crm_wing()

    interp = KRG()
    interp.set_training_values(xt, yt)
    with Silence():
        interp.train()
    return xt, yt, interp

def setup_QP():

    xt, yt, xlimits = get_rans_crm_wing()
    interp = QP() 
    interp.set_training_values(xt, yt)
    with Silence():
        interp.train()
    return xt, yt, interp



class ArrayOutputTest(SMTestCase):

    def test_QP(self):

        xt, yt, interp = setup_QP()
        with Silence():
            d0 = interp.predict_derivatives(np.atleast_2d(xt[10, :]), 0)

        self.assert_error(d0, np.array([[0.02588578, 5.86555448]]), atol=1e-6, rtol=1e-6)

    def test_KRG(self):

        xt, yt, interp = setup_KRG()
        with Silence():
            d0 = interp.predict_derivatives(np.atleast_2d(xt[10, :]), 0)

        self.assert_error(d0, np.array([[0.06874097, 5.26604232]]), atol=1e-6, rtol=1e-6)

if __name__ == '__main__':

    # krg = setup_KRG()
    # qp = setup_QP()
    unittest.main()
