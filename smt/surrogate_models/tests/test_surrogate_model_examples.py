"""
Author: John Hwang <<hwangjt@umich.edu>>

This package is distributed under New BSD license.
"""

import unittest

try:
    import matplotlib

    matplotlib.use("Agg")

    NO_MATPLOTLIB = False
except ImportError:
    NO_MATPLOTLIB = True

try:
    from smt.surrogate_models import IDW, RBF, RMTB, RMTC  # noqa: F401

    NO_COMPILED = False
except ImportError:
    NO_COMPILED = True

from smt.surrogate_models.gpx import GPX_AVAILABLE


class Test(unittest.TestCase):
    @unittest.skipIf(
        NO_COMPILED or NO_MATPLOTLIB, "C compilation failed or no matplotlib"
    )
    def test_idw(self):
        import matplotlib.pyplot as plt
        import numpy as np

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

    @unittest.skipIf(
        NO_COMPILED or NO_MATPLOTLIB, "C compilation failed or no matplotlib"
    )
    def test_rbf(self):
        import matplotlib.pyplot as plt
        import numpy as np

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

    @unittest.skipIf(
        NO_COMPILED or NO_MATPLOTLIB, "C compilation failed or no matplotlib"
    )
    def test_rmtb(self):
        import matplotlib.pyplot as plt
        import numpy as np

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

    @unittest.skipIf(
        NO_COMPILED or NO_MATPLOTLIB, "C compilation failed or no matplotlib"
    )
    def test_rmtc(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.surrogate_models import RMTC

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

        xlimits = np.array([[0.0, 4.0]])

        sm = RMTC(
            xlimits=xlimits,
            num_elements=6,
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

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_ls(self):
        import matplotlib.pyplot as plt
        import numpy as np

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

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_qp(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.surrogate_models import QP

        xt = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]).T
        yt = np.array([[0.2, 1.4, 1.5, 0.9, 1.0], [0.0, 1.0, 2.0, 4, 3]]).T

        sm = QP()
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)

        plt.plot(xt, yt[:, 0], "o", "C0")
        plt.plot(x, y[:, 0], "C0", label="Prediction 1")
        plt.plot(xt, yt[:, 1], "o", "C1")
        plt.plot(x, y[:, 1], "C1", label="Prediction 2")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_krg(self):
        import matplotlib.pyplot as plt
        import numpy as np

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
        _dydx = sm.predict_derivatives(xt, 0)
        _, axs = plt.subplots(1)

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

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_mixed_int_krg(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.applications.mixed_integer import MixedIntegerKrigingModel
        from smt.surrogate_models import KRG
        from smt.utils.design_space import HAS_SMTDesignSpace
        
        if HAS_SMTDesignSpace:
            from SMTDesignSpace.design_space import (
                DesignSpace,
                IntegerVariable
            )
        else:
            from smt.utils.design_space import (
                DesignSpace,
                IntegerVariable
            )
        xt = np.array([0.0, 2.0, 3.0])
        yt = np.array([0.0, 1.5, 0.9])

        design_space = DesignSpace(
            [
                IntegerVariable(0, 4),
            ]
        )
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(design_space=design_space, theta0=[1e-2], hyper_opt="Cobyla")
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

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_noisy_krg(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.surrogate_models import KRG

        # defining the toy example
        def target_fun(x):
            import numpy as np

            return np.cos(5 * x)

        nobs = 50  # number of obsertvations
        np.random.seed(0)  # a seed for reproducibility
        xt = np.random.uniform(size=nobs)  # design points

        # adding a random noise to observations
        yt = target_fun(xt) + np.random.normal(scale=0.05, size=nobs)

        # training the model with the option eval_noise= True
        sm = KRG(eval_noise=True, hyper_opt="Cobyla")
        sm.set_training_values(xt, yt)
        sm.train()

        # predictions
        x = np.linspace(0, 1, 100).reshape(-1, 1)
        y = sm.predict_values(x)  # predictive mean
        var = sm.predict_variances(x)  # predictive variance

        # plotting predictions +- 3 std confidence intervals
        plt.rcParams["figure.figsize"] = [8, 4]
        plt.fill_between(
            np.ravel(x),
            np.ravel(y - 3 * np.sqrt(var)),
            np.ravel(y + 3 * np.sqrt(var)),
            alpha=0.2,
            label="Confidence Interval 99%",
        )
        plt.scatter(xt, yt, label="Training noisy data")
        plt.plot(x, y, label="Prediction")
        plt.plot(x, target_fun(x), label="target function")
        plt.title("Kriging model with noisy observations")
        plt.legend(loc=0)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_mixed_gower_krg(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.applications.mixed_integer import (
            MixedIntegerKrigingModel,
        )
        from smt.surrogate_models import (
            KRG,
            CategoricalVariable,
            DesignSpace,
            MixIntKernelType,
        )

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
                hyper_opt="Cobyla",
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        y = sm.predict_values(x)

        plt.plot(xt, yt, "o", label="data")
        plt.plot(x, y, "d", color="red", markersize=3, label="pred")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    def test_kpls_auto(self):
        import numpy as np

        from smt.problems import TensorProduct
        from smt.sampling_methods import LHS
        from smt.surrogate_models import KPLS

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
        ncomp = sm.options["n_comp"]
        print("\n The model automatically choose " + str(ncomp) + " components.")

        ## You can predict a 10-dimension point from the 3-dimensional model
        print(
            sm.predict_values(
                np.array([[-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]])
            )
        )
        print(
            sm.predict_variances(
                np.array([[-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]])
            )
        )

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_kpls(self):
        import matplotlib.pyplot as plt
        import numpy as np

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
        _dydx = sm.predict_derivatives(xt, 0)

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

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_kplsk(self):
        import matplotlib.pyplot as plt
        import numpy as np

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
        _dydx = sm.predict_derivatives(xt, 0)

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

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_gekpls(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.problems import Sphere
        from smt.sampling_methods import LHS
        from smt.surrogate_models import GEKPLS, DesignSpace

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
                ).item()

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(X, Y, Z)

        plt.show()

    @unittest.skipIf(
        NO_MATPLOTLIB or not GPX_AVAILABLE, "Matplotlib or egobox not installed"
    )
    def test_gpx(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.surrogate_models import GPX

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

        sm = GPX(theta0=[1e-2])
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0.0, 4.0, num)
        y = sm.predict_values(x)
        # estimated variance
        s2 = sm.predict_variances(x)

        _, axs = plt.subplots(1)
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

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_mgp(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.sampling_methods import LHS
        from smt.surrogate_models import MGP

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
        upper = np.sum(np.abs(emb), axis=0).item()
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

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_sgp_fitc(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.surrogate_models import SGP

        def f_obj(x):
            import numpy as np

            return (
                np.sin(3 * np.pi * x)
                + 0.3 * np.cos(9 * np.pi * x)
                + 0.5 * np.sin(7 * np.pi * x)
            )

        # random generator for reproducibility
        rng = np.random.RandomState(0)

        # Generate training data
        nt = 200
        # Variance of the gaussian noise on our trainingg data
        eta2 = [0.01]
        gaussian_noise = rng.normal(loc=0.0, scale=np.sqrt(eta2), size=(nt, 1))
        xt = 2 * rng.rand(nt, 1) - 1
        yt = f_obj(xt) + gaussian_noise

        # Pick inducing points randomly in training data
        n_inducing = 30
        random_idx = rng.permutation(nt)[:n_inducing]
        Z = xt[random_idx].copy()

        sgp = SGP()
        sgp.set_training_values(xt, yt)
        sgp.set_inducing_inputs(Z=Z)
        # sgp.set_inducing_inputs()  # When Z not specified n_inducing points are picked randomly in traing data
        sgp.train()

        x = np.linspace(-1, 1, nt + 1).reshape(-1, 1)
        y = f_obj(x)
        hat_y = sgp.predict_values(x)
        var = sgp.predict_variances(x)

        # plot prediction
        plt.figure(figsize=(14, 6))
        plt.plot(x, y, "C1-", label="target function")
        plt.scatter(xt, yt, marker="o", s=10, label="observed data")
        plt.plot(x, hat_y, "k-", label="Sparse GP")
        plt.plot(x, hat_y - 3 * np.sqrt(var), "k--")
        plt.plot(x, hat_y + 3 * np.sqrt(var), "k--", label="99% CI")
        plt.plot(Z, -2.9 * np.ones_like(Z), "r|", mew=2, label="inducing points")
        plt.ylim([-3, 3])
        plt.legend(loc=0)
        plt.show()

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_sgp_vfe(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.surrogate_models import SGP

        def f_obj(x):
            import numpy as np

            return (
                np.sin(3 * np.pi * x)
                + 0.3 * np.cos(9 * np.pi * x)
                + 0.5 * np.sin(7 * np.pi * x)
            )

        # random generator for reproducibility
        rng = np.random.RandomState(42)

        # Generate training data
        nt = 200
        # Variance of the gaussian noise on our training data
        eta2 = [0.01]
        gaussian_noise = rng.normal(loc=0.0, scale=np.sqrt(eta2), size=(nt, 1))
        xt = 2 * rng.rand(nt, 1) - 1
        yt = f_obj(xt) + gaussian_noise

        # Pick inducing points randomly in training data
        n_inducing = 30
        random_idx = rng.permutation(nt)[:n_inducing]
        Z = xt[random_idx].copy()

        sgp = SGP(method="VFE")
        sgp.set_training_values(xt, yt)
        sgp.set_inducing_inputs(Z=Z)
        sgp.train()

        x = np.linspace(-1, 1, nt + 1).reshape(-1, 1)
        y = f_obj(x)
        hat_y = sgp.predict_values(x)
        var = sgp.predict_variances(x)

        # plot prediction
        plt.figure(figsize=(14, 6))
        plt.plot(x, y, "C1-", label="target function")
        plt.scatter(xt, yt, marker="o", s=10, label="observed data")
        plt.plot(x, hat_y, "k-", label="Sparse GP")
        plt.plot(x, hat_y - 3 * np.sqrt(var), "k--")
        plt.plot(x, hat_y + 3 * np.sqrt(var), "k--", label="99% CI")
        plt.plot(Z, -2.9 * np.ones_like(Z), "r|", mew=2, label="inducing points")
        plt.ylim([-3, 3])
        plt.legend(loc=0)
        plt.show()

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_genn(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.surrogate_models import GENN

        # Test function
        def f(x):
            import numpy as np  # need to repeat for sphinx_auto_embed

            return x * np.sin(x)

        def df_dx(x):
            import numpy as np  # need to repeat for sphinx_auto_embed

            return np.sin(x) + x * np.cos(x)

        # Domain
        lb = -np.pi
        ub = np.pi

        # Training data
        m = 4
        xt = np.linspace(lb, ub, m)
        yt = f(xt)
        dyt_dxt = df_dx(xt)

        # Validation data
        xv = lb + np.random.rand(30, 1) * (ub - lb)
        yv = f(xv)
        # dyv_dxv = df_dx(xv)

        # Instantiate
        genn = GENN()

        # Likely the only options a user will interact with
        genn.options["hidden_layer_sizes"] = [6, 6]
        genn.options["alpha"] = 0.1
        genn.options["lambd"] = 0.1
        genn.options["gamma"] = (
            1.0  # 1 = gradient-enhanced on, 0 = gradient-enhanced off
        )
        genn.options["num_iterations"] = 1000
        genn.options["is_backtracking"] = True
        genn.options["is_normalize"] = False

        # Train
        genn.load_data(xt, yt, dyt_dxt)
        genn.train()

        # Plot comparison
        if genn.options["gamma"] == 1.0:
            title = "with gradient enhancement"
        else:
            title = "without gradient enhancement"
        x = np.arange(lb, ub, 0.01)
        y = f(x)
        y_pred = genn.predict_values(x)
        fig, ax = plt.subplots()
        ax.plot(x, y_pred)
        ax.plot(x, y, "k--")
        ax.plot(xv, yv, "ro")
        ax.plot(xt, yt, "k+", mew=3, ms=10)
        ax.set(xlabel="x", ylabel="y", title=title)
        ax.legend(["Predicted", "True", "Test", "Train"])
        plt.show()


if __name__ == "__main__":
    unittest.main()
