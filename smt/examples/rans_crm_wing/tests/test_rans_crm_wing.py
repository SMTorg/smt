import unittest

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

try:
    from smt.surrogate_models import RMTB, RMTC

    compiled_available = True
except:
    compiled_available = False
from smt.examples.rans_crm_wing.rans_crm_wing import (
    get_rans_crm_wing,
    plot_rans_crm_wing,
)


class Test(unittest.TestCase):
    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtb(self):
        from smt.surrogate_models import RMTB

        xt, yt, xlimits = get_rans_crm_wing()

        interp = RMTB(
            num_ctrl_pts=20,
            xspecs={"xlimits": xlimits},
            nonlinear_maxiter=100,
            energy_weight=1e-12,
        )
        interp.set_training_values(xt, yt)
        interp.train()

        plot_rans_crm_wing(xt, yt, xlimits, interp)

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtc(self):
        from smt.surrogate_models import RMTC

        xt, yt, xlimits = get_rans_crm_wing()

        interp = RMTC(
            num_elements=20,
            xspecs={"xlimits": xlimits},
            nonlinear_maxiter=100,
            energy_weight=1e-10,
        )
        interp.set_training_values(xt, yt)
        interp.train()

        plot_rans_crm_wing(xt, yt, xlimits, interp)


if __name__ == "__main__":
    Test().test_rmtb()
    plt.savefig("testRMTB.pdf")
    Test().test_rmtc()
    plt.savefig("testRMTC.pdf")
