import unittest

import matplotlib
matplotlib.use('Agg')

try:
    from smt.methods import RMTB, RMTC
    compiled_available = True
except:
    compiled_available = False


class Test(unittest.TestCase):

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtb(self):
        from smt.methods import RMTB
        from smt.examples.one_d_step import get_one_d_step, plot_one_d_step

        xt, yt, xlimits = get_one_d_step()

        interp = RMTB(num_ctrl_pts=100, xlimits=xlimits, nln_max_iter=20, reg_cons=1e-14)
        interp.set_training_values(xt, yt)
        interp.train()

        plot_one_d_step(xt, yt, xlimits, interp)

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtc(self):
        from smt.methods import RMTC
        from smt.examples.one_d_step import get_one_d_step, plot_one_d_step

        xt, yt, xlimits = get_one_d_step()

        interp = RMTC(num_elements=40, xlimits=xlimits, nln_max_iter=20, reg_cons=1e-14)
        interp.set_training_values(xt, yt)
        interp.train()

        plot_one_d_step(xt, yt, xlimits, interp)
