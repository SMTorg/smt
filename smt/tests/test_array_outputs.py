import numpy as np
import unittest

from smt.utils.sm_test_case import SMTestCase
from smt.utils.silence import Silence
from smt.surrogate_models import QP, KRG
from smt.examples.rans_crm_wing.rans_crm_wing import (
    get_rans_crm_wing,
    plot_rans_crm_wing,
)


def setup_sm(sm_name, settings={}):
    xt, yt, xlimits = get_rans_crm_wing()

    _tmp = __import__("smt", globals(), locals(), ["surrogate_models"], 0)
    interp = getattr(_tmp.surrogate_models, sm_name)(**settings)

    interp.set_training_values(xt, yt)
    with Silence():
        interp.train()
    return xt, yt, interp


class ArrayOutputTest(SMTestCase):
    def test_QP(self):
        xt, yt, interp = setup_sm(sm_name="QP")
        with Silence():
            d0 = interp.predict_derivatives(np.atleast_2d(xt[10, :]), 0)

        self.assert_error(
            d0, np.array([[0.02588578, 5.86555448]]), atol=1e-6, rtol=1e-6
        )

    def test_KRG(self):
        xt, yt, interp = setup_sm(sm_name="KRG")
        with Silence():
            d0 = interp.predict_derivatives(np.atleast_2d(xt[10, :]), 0)

        self.assert_error(
            d0, np.array([[0.06874097, 4.366292277996716]]), atol=0.55, rtol=0.15
        )

    def test_RBF(self):
        xt, yt, interp = setup_sm(sm_name="RBF")
        with Silence():
            d0 = interp.predict_derivatives(np.atleast_2d(xt[10, :]), 0)

        self.assert_error(d0, np.array([[0.15741522, 4.80265154]]), atol=0.2, rtol=0.03)

    def test_LS(self):
        xt, yt, interp = setup_sm(sm_name="LS")
        with Silence():
            d0 = interp.predict_derivatives(np.atleast_2d(xt[10, :]), 0)

        self.assert_error(d0, np.array([[0.2912748, 5.39911101]]), atol=0.2, rtol=0.03)

    def test_IDW(self):
        xt, yt, interp = setup_sm(sm_name="IDW")
        with Silence():
            d0 = interp.predict_derivatives(np.atleast_2d(xt[10, :]), 0)

        self.assert_error(d0, np.array([[0.0, 0.0]]), atol=0.2, rtol=0.03)


if __name__ == "__main__":
    xt, yt, sm = setup_sm("QP")
    unittest.main()
