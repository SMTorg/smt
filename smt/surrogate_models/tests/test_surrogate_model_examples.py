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
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

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
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

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
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

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
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

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
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

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
        yt = np.array([[0.2, 1.4, 1.5, 0.9, 1.0], [0.0, 1.0, 2.0, 4, 3]]).T

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
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

        sm = KRG(theta0=[1e-2])
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)
        # estimated variance
        s2 = sm.predict_variances(x)
        # derivative according to the first variable
        dydx = sm.predict_derivatives(xt, 0)
        fig, axs = plt.subplots(1)

        # add a plot with variance
        axs.plot(xt, yt, "o")
        axs.plot(x, y)
        axs.fill_between(
            np.ravel(x),
            np.ravel(y - 3 * np.sqrt(s2)),
            np.ravel(y + 3 * np.sqrt(s2)),
            color="lightgrey",
        )
        axs.set_xlabel("x")
        axs.set_ylabel("y")
        axs.legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="lower right",
        )

        plt.show()

    def test_mixed_int_krg(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import KRG
        from smt.applications.mixed_integer import MixedIntegerKrigingModel
        from smt.utils.design_space import DesignSpace, IntegerVariable

        xt = np.array([0.0, 2.0, 3.0])
        yt = np.array([0.0, 1.5, 0.9])

        design_space = DesignSpace(
            [
                IntegerVariable(0, 4),
            ]
        )
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(design_space=design_space, theta0=[1e-2])
        )
        sm.set_training_values(xt, yt)
        sm.train()

        num = 500
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)
        # estimated variance
        s2 = sm.predict_variances(x)

        fig, axs = plt.subplots(1)
        axs.plot(xt, yt, "o")
        axs.plot(x, y)
        axs.fill_between(
            np.ravel(x),
            np.ravel(y - 3 * np.sqrt(s2)),
            np.ravel(y + 3 * np.sqrt(s2)),
            color="lightgrey",
        )
        axs.set_xlabel("x")
        axs.set_ylabel("y")
        axs.legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="lower right",
        )

        plt.show()

    def test_mixed_gower_krg(self):
        from smt.surrogate_models import (
            MixIntKernelType,
            KRG,
            DesignSpace,
            CategoricalVariable,
        )
        from smt.applications.mixed_integer import (
            MixedIntegerKrigingModel,
        )
        import matplotlib.pyplot as plt
        import numpy as np

        xt = np.array([0, 3, 4])
        yt = np.array([0.0, 1.0, 1.5])
        design_space = DesignSpace(
            [
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
            ]
        )

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=design_space,
                theta0=[1e-2],
                categorical_kernel=MixIntKernelType.GOWER,
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 5, 5)
        y = sm.predict_values(x)

        plt.plot(xt, yt, "o", label="data")
        plt.plot(x, y, "d", color="red", markersize=3, label="pred")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    def test_kpls_auto(self):
        import numpy as np
        from smt.surrogate_models import KPLS
        from smt.problems import TensorProduct
        from smt.sampling_methods import LHS

        # The problem is the exponential problem with dimension 10
        ndim = 10
        prob = TensorProduct(ndim=ndim, func="exp")

        sm = KPLS(eval_n_comp=True)
        samp = LHS(xlimits=prob.xlimits, random_state=42)
        np.random.seed(0)
        xt = samp(50)
        yt = prob(xt)
        np.random.seed(1)
        sm.set_training_values(xt, yt)
        sm.train()

        ## The model automatically choose a dimension of 3
        l = sm.options["n_comp"]
        print("\n The model automatically choose " + str(l) + " components.")

        ## You can predict a 10-dimension point from the 3-dimensional model
        print(sm.predict_values(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])))
        print(sm.predict_variances(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])))

    def test_kpls(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import KPLS

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

        sm = KPLS(theta0=[1e-2])
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)
        # estimated variance
        # add a plot with variance
        s2 = sm.predict_variances(x)
        # to compute the derivative according to the first variable
        dydx = sm.predict_derivatives(xt, 0)

        plt.plot(xt, yt, "o")
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Training data", "Prediction"])
        plt.show()

        plt.plot(xt, yt, "o")
        plt.plot(x, y)
        plt.fill_between(
            np.ravel(x),
            np.ravel(y - 3 * np.sqrt(s2)),
            np.ravel(y + 3 * np.sqrt(s2)),
            color="lightgrey",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Training data", "Prediction", "Confidence Interval 99%"])
        plt.show()

    def test_kplsk(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import KPLSK

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

        sm = KPLSK(theta0=[1e-2])
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)
        # estimated variance
        s2 = sm.predict_variances(x)
        # derivative according to the first variable
        dydx = sm.predict_derivatives(xt, 0)

        plt.plot(xt, yt, "o")
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Training data", "Prediction"])
        plt.show()

        # add a plot with variance
        plt.plot(xt, yt, "o")
        plt.plot(x, y)
        plt.fill_between(
            np.ravel(x),
            np.ravel(y - 3 * np.sqrt(s2)),
            np.ravel(y + 3 * np.sqrt(s2)),
            color="lightgrey",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Training data", "Prediction", "Confidence Interval 99%"])
        plt.show()

    def test_gekpls(self):
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        from smt.surrogate_models import GEKPLS, DesignSpace
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
        design_space = DesignSpace(fun.xlimits)
        # Build the GEKPLS model
        n_comp = 2
        sm = GEKPLS(
            design_space=design_space,
            theta0=[1e-2] * n_comp,
            extra_points=1,
            print_prediction=False,
            n_comp=n_comp,
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
        ax = fig.add_subplot(projection="3d")
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

    def test_mgp(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from smt.surrogate_models import MGP
        from smt.sampling_methods import LHS

        # Construction of the DOE
        dim = 3

        def fun(x):
            import numpy as np

            res = (
                np.sum(x, axis=1) ** 2
                - np.sum(x, axis=1)
                + 0.2 * (np.sum(x, axis=1) * 1.2) ** 3
            )
            return res

        sampling = LHS(
            xlimits=np.asarray([(-1, 1)] * dim), criterion="m", random_state=42
        )
        xt = sampling(8)
        yt = np.atleast_2d(fun(xt)).T

        # Build the MGP model
        sm = MGP(
            theta0=[1e-2],
            print_prediction=False,
            n_comp=1,
        )
        sm.set_training_values(xt, yt[:, 0])
        sm.train()

        # Get the transfert matrix A
        emb = sm.embedding["C"]

        # Compute the smallest box containing all points of A
        upper = np.sum(np.abs(emb), axis=0)
        lower = -upper

        # Test the model
        u_plot = np.atleast_2d(np.arange(lower, upper, 0.01)).T
        x_plot = sm.get_x_from_u(u_plot)  # Get corresponding points in Omega
        y_plot_true = fun(x_plot)
        y_plot_pred = sm.predict_values(u_plot)
        sigma_MGP = sm.predict_variances(u_plot)
        sigma_KRG = sm.predict_variances_no_uq(u_plot)

        u_train = sm.get_u_from_x(xt)  # Get corresponding points in A

        # Plots
        fig, ax = plt.subplots()
        ax.plot(u_plot, y_plot_pred, label="Predicted")
        ax.plot(u_plot, y_plot_true, "k--", label="True")
        ax.plot(u_train, yt, "k+", mew=3, ms=10, label="Train")
        ax.fill_between(
            u_plot[:, 0],
            y_plot_pred[:, 0] - 3 * sigma_MGP[:, 0],
            y_plot_pred[:, 0] + 3 * sigma_MGP[:, 0],
            color="r",
            alpha=0.5,
            label="Variance with hyperparameters uncertainty",
        )
        ax.fill_between(
            u_plot[:, 0],
            y_plot_pred[:, 0] - 3 * sigma_KRG[:, 0],
            y_plot_pred[:, 0] + 3 * sigma_KRG[:, 0],
            color="b",
            alpha=0.5,
            label="Variance without hyperparameters uncertainty",
        )

        ax.set(xlabel="x", ylabel="y", title="MGP")
        fig.legend(loc="upper center", ncol=2)
        fig.tight_layout()
        fig.subplots_adjust(top=0.74)
        plt.show()


if __name__ == "__main__":
    unittest.main()
