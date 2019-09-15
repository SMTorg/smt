"""
Author: Remi Lafage <remi.lafage@onera.fr> and Nathalie Bartoli
This package is distributed under New BSD license.
"""
import unittest
import numpy as np
from sys import argv
import matplotlib

# matplotlib.use("Agg")

from smt.applications import EGO
from smt.utils.sm_test_case import SMTestCase
from smt.problems import Branin, Rosenbrock
from smt.sampling_methods import FullFactorial


class TestEGO(SMTestCase):
    """
    Test class
    """

    plot = None

    @staticmethod
    def function_test_1d(x):
        # function xsinx
        x = np.reshape(x, (-1,))
        y = np.zeros(x.shape)
        y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
        return y.reshape((-1, 1))

    def test_function_test_1d(self):
        n_iter = 15
        xlimits = np.array([[0.0, 25.0]])

        criterion = "EI"

        ego = EGO(n_iter=n_iter, criterion=criterion, n_doe=3, xlimits=xlimits)

        x_opt, y_opt, _, _, _, _, _ = ego.optimize(fun=TestEGO.function_test_1d)
        print(x_opt, y_opt)

        self.assertAlmostEqual(18.9, float(x_opt), delta=1)
        self.assertAlmostEqual(-15.1, float(y_opt), delta=1)

    def test_rosenbrock_2D(self):
        n_iter = 30
        fun = Rosenbrock(ndim=2)
        xlimits = fun.xlimits
        criterion = "UCB"  #'EI' or 'SBO' or 'UCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        ego = EGO(xdoe=xdoe, n_iter=n_iter, criterion=criterion, xlimits=xlimits)

        x_opt, y_opt, _, _, _, _, _ = ego.optimize(fun=fun)

        self.assertTrue(np.allclose([[1, 1]], x_opt, rtol=0.5))
        self.assertAlmostEqual(0.0, float(y_opt), delta=1)

    def test_branin_2D(self):
        n_iter = 10
        fun = Branin(ndim=2)
        xlimits = fun.xlimits
        criterion = "UCB"  #'EI' or 'SBO' or 'UCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        ego = EGO(xdoe=xdoe, n_iter=n_iter, criterion=criterion, xlimits=xlimits)

        x_opt, y_opt, _, _, _, _, _ = ego.optimize(fun=fun)

        # 3 optimal points possible: [-pi,12.275], [pi, 12.275], [9.42478,2.475]
        self.assertTrue(
            np.allclose([[-3.14, 12.275]], x_opt, rtol=0.1)
            or np.allclose([[3.14, 12.275]], x_opt, rtol=0.1)
            or np.allclose([[9.42, 2.475]], x_opt, rtol=0.1)
        )
        self.assertAlmostEqual(0.39, float(y_opt), delta=1)

    @staticmethod
    def run_ego_example():
        import numpy as np
        import six
        from smt.applications import EGO
        from smt.sampling_methods import FullFactorial
        from smt.surrogate_models import KRG

        import sklearn
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.stats import norm

        def EI(t, points, f_min):
            pred = t.predict_values(points)
            var = t.predict_variances(points)
            args0 = (f_min - pred) / var
            args1 = (f_min - pred) * norm.cdf(args0)
            args2 = var * norm.pdf(args0)
            ei = args1 + args2
            return ei

        def function_test_1d(x):
            # function xsinx
            import numpy as np

            x = np.reshape(x, (-1,))
            y = np.zeros(x.shape)
            y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
            return y.reshape((-1, 1))

        n_iter = 8
        xlimits = np.array([[0.0, 25.0]])
        xdoe = np.atleast_2d([0, 7, 25]).T

        criterion = "EI"  #'EI' or 'SBO' or 'UCB'

        ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits)

        x_opt, y_opt, ind_best, x_data, y_data, x_doe, y_doe = ego.optimize(
            fun=function_test_1d
        )
        print(x_opt, y_opt)
        # Check if the optimal point is Xopt=18.9, Yopt =-15.1

        X_plot = np.atleast_2d(np.linspace(0, 25, 100)).T
        Y_plot = function_test_1d(X_plot)
        gpr = KRG()
        gpr.options["print_global"] = False

        fig = plt.figure(figsize=[10, 10])
        for k in range(n_iter):
            x_data_k = x_data[0 : k + 3]
            y_data_k = y_data[0 : k + 3]
            gpr.set_training_values(x_data_k, y_data_k)
            gpr.train()
            obj_k = lambda x: -EI(gpr, np.atleast_2d(x), np.min(y_data_k))

            Y_GP_plot = gpr.predict_values(X_plot)
            Y_GP_plot_var = gpr.predict_variances(X_plot)
            Y_EI_plot = obj_k(X_plot)

            ax = fig.add_subplot(4, 2, k + 1)
            ax1 = ax.twinx()
            ei, = ax1.plot(X_plot, Y_EI_plot, color="red")

            true_fun, = ax.plot(X_plot, Y_plot)
            data, = ax.plot(
                x_data_k, y_data_k, linestyle="", marker="o", color="orange"
            )
            if k + 4 < n_iter - 1:
                opt, = ax.plot(
                    x_data[k + 3], y_data[k + 3], linestyle="", marker="*", color="r"
                )
            gp, = ax.plot(X_plot, Y_GP_plot, linestyle="--", color="g")
            sig_plus = Y_GP_plot + 3 * Y_GP_plot_var
            sig_moins = Y_GP_plot - 3 * Y_GP_plot_var
            un_gp = ax.fill_between(
                X_plot.T[0], sig_plus.T[0], sig_moins.T[0], alpha=0.3, color="g"
            )
            lines = [true_fun, data, gp, un_gp, opt, ei]
            fig.suptitle("EGO optimization of $x \sin{x}$ function")
            fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.8)
            ax.set_title("iteration {}".format(k))
            fig.legend(
                lines,
                [
                    "True function",
                    "Data",
                    "GPR prediction",
                    "99 % confidence",
                    "Next point to Evaluate",
                    "Infill Criteria",
                ],
            )
            # plt.savefig("Optimisation_%d" % k)
            # plt.close(fig)
        plt.show()


if __name__ == "__main__":
    if "--plot" in argv:
        TestEGO.plot = True
        argv.remove("--plot")
    if "--example" in argv:
        TestEGO.run_ego_example()
        exit()
    unittest.main()

