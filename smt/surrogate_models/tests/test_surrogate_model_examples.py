"""
Author: John Hwang <<hwangjt@umich.edu>>

This package is distributed under New BSD license.
"""

import unittest

import matplotlib

matplotlib.use("Agg")

try:
    from smt.surrogate_models import IDW, RBF, RMTB, RMTC

    compiled_available = True
except:
    compiled_available = False


class Test(unittest.TestCase):
    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_idw(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import IDW

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])

        sm = IDW(p=2)
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, "o")
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Training data", "Prediction"])
        plt.show()

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rbf(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import RBF

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])

        sm = RBF(d0=5)
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, "o")
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Training data", "Prediction"])
        plt.show()

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtb(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import RMTB

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])

        xlimits = np.array([[0.0, 4.0]])

        sm = RMTB(
            xlimits=xlimits,
            order=4,
            num_ctrl_pts=20,
            energy_weight=1e-15,
            regularization_weight=0.0,
        )
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, "o")
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Training data", "Prediction"])
        plt.show()

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtc(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import RMTC

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])

        xlimits = np.array([[0.0, 4.0]])

        sm = RMTC(
            xlimits=xlimits,
            num_elements=20,
            energy_weight=1e-15,
            regularization_weight=0.0,
        )
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, "o")
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Training data", "Prediction"])
        plt.show()

    def test_ls(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import LS

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])

        sm = LS()
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, "o")
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Training data", "Prediction"])
        plt.show()

    def test_qp(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import QP

        xt = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]).T
        yt = np.array([[0.2, 1.4, 1.5, 0.5, 1.0], [0.0, 1.0, 2.0, 4, 3]]).T

        sm = QP()
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)

        t1, _ = plt.plot(xt, yt[:, 0], "o", "C0")
        p1 = plt.plot(x, y[:, 0], "C0", label="Prediction 1")
        t2, _ = plt.plot(xt, yt[:, 1], "o", "C1")
        p2 = plt.plot(x, y[:, 1], "C1", label="Prediction 2")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    def test_krg(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import KRG

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])

        sm = KRG(theta0=[1e-2])
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, "o")
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Training data", "Prediction"])
        plt.show()

    def test_kpls(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import KPLS

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])

        sm = KPLS(theta0=[1e-2])
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, "o")
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Training data", "Prediction"])
        plt.show()

    def test_kplsk(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import KPLSK

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])

        sm = KPLSK(theta0=[1e-2])
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)
        yy = sm.predict_derivatives(xt, 0)
        plt.plot(xt, yt, "o")
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Training data", "Prediction"])
        plt.show()

    def test_gekpls(self):
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        from smt.surrogate_models import GEKPLS
        from smt.problems import Sphere
        from smt.sampling_methods import LHS

        # Construction of the DOE
        fun = Sphere(ndim=2)
        sampling = LHS(xlimits=fun.xlimits, criterion="m")
        xt = sampling(20)
        yt = fun(xt)
        # Compute the gradient
        for i in range(2):
            yd = fun(xt, kx=i)
            yt = np.concatenate((yt, yd), axis=1)

        # Build the GEKPLS model
        sm = GEKPLS(
            theta0=[1e-2], xlimits=fun.xlimits, extra_points=1, print_prediction=False
        )
        sm.set_training_values(xt, yt[:, 0])
        for i in range(2):
            sm.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)
        sm.train()

        # Test the model
        X = np.arange(fun.xlimits[0, 0], fun.xlimits[0, 1], 0.25)
        Y = np.arange(fun.xlimits[1, 0], fun.xlimits[1, 1], 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros((X.shape[0], X.shape[1]))

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = sm.predict_values(
                    np.hstack((X[i, j], Y[i, j])).reshape((1, 2))
                )

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        surf = ax.plot_surface(X, Y, Z)

        plt.show()

    def test_genn(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from smt.surrogate_models.genn import GENN, load_smt_data

        # Training data
        lower_bound = -np.pi
        upper_bound = np.pi
        number_of_training_points = 4
        xt = np.linspace(lower_bound, upper_bound, number_of_training_points)
        yt = xt * np.sin(xt)
        dyt_dxt = np.sin(xt) + xt * np.cos(xt)

        # Validation data
        number_of_validation_points = 30
        xv = np.linspace(lower_bound, upper_bound, number_of_validation_points)
        yv = xv * np.sin(xv)
        dyv_dxv = np.sin(xv) + xv * np.cos(xv)

        # Truth model
        x = np.arange(lower_bound, upper_bound, 0.01)
        y = x * np.sin(x)

        # GENN
        genn = GENN()
        genn.options["alpha"] = 0.1  # learning rate that controls optimizer step size
        genn.options["beta1"] = 0.9  # tuning parameter to control ADAM optimization
        genn.options["beta2"] = 0.99  # tuning parameter to control ADAM optimization
        genn.options[
            "lambd"
        ] = 0.1  # lambd = 0. = no regularization, lambd > 0 = regularization
        genn.options[
            "gamma"
        ] = 1.0  # gamma = 0. = no grad-enhancement, gamma > 0 = grad-enhancement
        genn.options["deep"] = 2  # number of hidden layers
        genn.options["wide"] = 6  # number of nodes per hidden layer
        genn.options[
            "mini_batch_size"
        ] = 64  # used to divide data into training batches (use for large data sets)
        genn.options["num_epochs"] = 20  # number of passes through data
        genn.options[
            "num_iterations"
        ] = 100  # number of optimizer iterations per mini-batch
        genn.options["is_print"] = True  # print output (or not)
        load_smt_data(
            genn, xt, yt, dyt_dxt
        )  # convenience function to read in data that is in SMT format
        genn.train()  # API function to train model
        genn.plot_training_history()  # non-API function to plot training history (to check convergence)
        genn.goodness_of_fit(
            xv, yv, dyv_dxv
        )  # non-API function to check accuracy of regression
        y_pred = genn.predict_values(
            x
        )  # API function to predict values at new (unseen) points

        # Plot
        fig, ax = plt.subplots()
        ax.plot(x, y_pred)
        ax.plot(x, y, "k--")
        ax.plot(xv, yv, "ro")
        ax.plot(xt, yt, "k+", mew=3, ms=10)
        ax.set(xlabel="x", ylabel="y", title="GENN")
        ax.legend(["Predicted", "True", "Test", "Train"])
        plt.show()


if __name__ == "__main__":
    unittest.main()
