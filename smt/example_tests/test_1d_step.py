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

        xt = np.array([
            0.0000,    0.4000,    0.6000,    0.7000,    0.7500,
            0.7750,    0.8000,    0.8500,    0.8750,    0.9000,
            0.9250,    0.9500,    0.9750,    1.0000,    1.0250,
            1.0500,    1.1000,    1.2000,    1.3000,    1.4000,
            1.6000,    1.8000,    2.0000,
        ], dtype=np.float64)
        yt = np.array([
            0.0130,     0.0130,     0.0130,     0.0130,   0.0130,
            0.0130,     0.0130,     0.0132,     0.0135,   0.0140,
            0.0162,     0.0230,     0.0275,     0.0310,   0.0344,
            0.0366,     0.0396,     0.0410,     0.0403,   0.0390,
            0.0360,     0.0350,     0.0345,
        ], dtype=np.float64)

        xlimits = np.array([[0.0, 2.0]])

        interp = RMTB(num_ctrl_pts=100, xlimits=xlimits, nln_max_iter=20, reg_cons=1e-14)

        interp.set_training_values(xt, yt)
        interp.train()

        num = 500

        x = np.linspace(0., 2., num)
        y = interp.predict_values(x)[:, 0]

        plt.plot(x, y)
        plt.plot(xt, yt, 'o')
        plt.show()

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtc(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import RMTC

        xt = np.array([
            0.0000,    0.4000,    0.6000,    0.7000,    0.7500,
            0.7750,    0.8000,    0.8500,    0.8750,    0.9000,
            0.9250,    0.9500,    0.9750,    1.0000,    1.0250,
            1.0500,    1.1000,    1.2000,    1.3000,    1.4000,
            1.6000,    1.8000,    2.0000,
        ], dtype=np.float64)
        yt = np.array([
            0.0130,     0.0130,     0.0130,     0.0130,   0.0130,
            0.0130,     0.0130,     0.0132,     0.0135,   0.0140,
            0.0162,     0.0230,     0.0275,     0.0310,   0.0344,
            0.0366,     0.0396,     0.0410,     0.0403,   0.0390,
            0.0360,     0.0350,     0.0345,
        ], dtype=np.float64)

        xlimits = np.array([[0.0, 2.0]])

        interp = RMTC(num_elements=40, xlimits=xlimits, nln_max_iter=20, reg_cons=1e-14)

        interp.set_training_values(xt, yt)
        interp.train()

        num = 500

        x = np.linspace(0., 2., num)
        y = interp.predict_values(x)[:, 0]

        plt.plot(x, y)
        plt.plot(xt, yt, 'o')
        plt.show()
