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
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import RMTB
        from smt.examples.one_d_step import get_one_d_step

        xt, yt, xlimits = get_one_d_step()

        interp = RMTB(num_ctrl_pts=100, xlimits=xlimits, nln_max_iter=20, reg_cons=1e-14)
        interp.set_training_values(xt, yt)
        interp.train()

        num = 500
        x = np.linspace(0., 2., num)
        y = interp.predict_values(x)[:, 0]

        plt.plot(x, y)
        plt.plot(xt, yt, 'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtc(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import RMTC
        from smt.examples.one_d_step import get_one_d_step

        xt, yt, xlimits = get_one_d_step()

        interp = RMTC(num_elements=40, xlimits=xlimits, nln_max_iter=20, reg_cons=1e-14)
        interp.set_training_values(xt, yt)
        interp.train()

        num = 500
        x = np.linspace(0., 2., num)
        y = interp.predict_values(x)[:, 0]

        plt.plot(x, y)
        plt.plot(xt, yt, 'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
