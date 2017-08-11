import unittest

import matplotlib
matplotlib.use('Agg')

try:
    from smt.methods import IDW, RBF, RMTB, RMTC
    compiled_available = True
except:
    compiled_available = False


class Test(unittest.TestCase):

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_idw(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import IDW

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        sm = IDW(p=2)
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rbf(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import RBF

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        sm = RBF(d0=5)
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtb(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import RMTB

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        xlimits = np.array([[0., 4.]])

        sm = RMTB(xlimits=xlimits, order=4, num_ctrl_pts=20, reg_dv=1e-15, reg_cons=1e-15)
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtc(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import RMTC

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        xlimits = np.array([[0., 4.]])

        sm = RMTC(xlimits=xlimits, num_elements=20, reg_dv=1e-15, reg_cons=1e-15)
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    def test_ls(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import LS

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        sm = LS()
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    def test_qp(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import QP

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        sm = QP()
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    def test_krg(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import KRG

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        sm = KRG(theta0=[1e-2])
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()


if __name__ == '__main__':
    unittest.main()
