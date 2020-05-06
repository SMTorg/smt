"""
Author: Remi Lafage <remi.lafage@onera.fr> and Nathalie Bartoli
This package is distributed under New BSD license.
"""

import warnings

warnings.filterwarnings("ignore")

import unittest
import numpy as np
from sys import argv
import matplotlib

matplotlib.use("Agg")

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

    def test_function_test_1d_parallel(self):
        n_iter = 3
        xlimits = np.array([[0.0, 25.0]])

        criterion = "EI"
        n_par = 3
        ego = EGO(
            n_iter=n_iter, criterion=criterion, n_doe=3, xlimits=xlimits, n_par=n_par
        )
        x_opt, y_opt, _, _, _, _, _ = ego.optimize(fun=TestEGO.function_test_1d)

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

    def test_rosenbrock_2D_parallel(self):
        n_iter = 15
        n_par = 5
        fun = Rosenbrock(ndim=2)
        xlimits = fun.xlimits
        criterion = "UCB"  #'EI' or 'SBO' or 'UCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        qEIAproxCrit = "KB"
        ego = EGO(
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            xlimits=xlimits,
            n_par=n_par,
            qEIAproxCrit=qEIAproxCrit,
        )

        x_opt, y_opt, _, _, _, _, _ = ego.optimize(fun=fun)
        print("Rosenbrock: ", x_opt)
        self.assertTrue(np.allclose([[1, 1]], x_opt, rtol=0.5))
        self.assertAlmostEqual(0.0, float(y_opt), delta=1)

    def test_branin_2D(self):
        n_iter = 15
        fun = Branin(ndim=2)
        xlimits = fun.xlimits
        criterion = "UCB"  #'EI' or 'SBO' or 'UCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        ego = EGO(xdoe=xdoe, n_iter=n_iter, criterion=criterion, xlimits=xlimits)

        x_opt, y_opt, _, _, _, _, _ = ego.optimize(fun=fun)

        # 3 optimal points possible: [-pi, 12.275], [pi, 2.275], [9.42478, 2.475]
        print(x_opt)
        self.assertTrue(
            np.allclose([[-3.14, 12.275]], x_opt, rtol=0.1)
            or np.allclose([[3.14, 2.275]], x_opt, rtol=0.1)
            or np.allclose([[9.42, 2.475]], x_opt, rtol=0.1)
        )
        self.assertAlmostEqual(0.39, float(y_opt), delta=1)

    def test_branin_2D_parallel(self):
        n_iter = 10
        fun = Branin(ndim=2)
        n_par = 5
        xlimits = fun.xlimits
        criterion = "EI"  #'EI' or 'SBO' or 'UCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        ego = EGO(
            xdoe=xdoe, n_iter=n_iter, criterion=criterion, xlimits=xlimits, n_par=n_par
        )

        x_opt, y_opt, _, _, _, _, _ = ego.optimize(fun=fun)

        # 3 optimal points possible: [-pi, 12.275], [pi, 2.275], [9.42478, 2.475]
        print(x_opt)
        self.assertTrue(
            np.allclose([[-3.14, 12.275]], x_opt, rtol=0.1)
            or np.allclose([[3.14, 2.275]], x_opt, rtol=0.1)
            or np.allclose([[9.42, 2.475]], x_opt, rtol=0.1)
        )
        print("Branin=", x_opt)
        self.assertAlmostEqual(0.39, float(y_opt), delta=1)

    def test_ydoe_option(self):
        n_iter = 10
        fun = Branin(ndim=2)
        xlimits = fun.xlimits
        criterion = "UCB"  #'EI' or 'SBO' or 'UCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        ydoe = fun(xdoe)

        ego = EGO(
            xdoe=xdoe, ydoe=ydoe, n_iter=n_iter, criterion=criterion, xlimits=xlimits
        )
        _, y_opt, _, _, _, _, y_doe = ego.optimize(fun=fun)

        self.assertAlmostEqual(0.39, float(y_opt), delta=1)

    def test_find_points(self):
        fun = TestEGO.function_test_1d
        xlimits = np.array([[0.0, 25.0]])
        xdoe = FullFactorial(xlimits=xlimits)(3)
        ydoe = fun(xdoe)
        ego = EGO(
            xdoe=xdoe, ydoe=ydoe, n_iter=1, criterion="UCB", xlimits=xlimits, n_start=30
        )
        _, _, _, _, _, _, _ = ego.optimize(fun=fun)
        x, _ = ego._find_points(xdoe, ydoe)
        self.assertAlmostEqual(6.5, float(x), delta=1)

    @staticmethod
    def run_ego_example():
        import numpy as np
        import six
        from smt.applications import EGO
        from smt.sampling_methods import FullFactorial

        import sklearn
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.stats import norm

        def function_test_1d(x):
            # function xsinx
            import numpy as np

            x = np.reshape(x, (-1,))
            y = np.zeros(x.shape)
            y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
            return y.reshape((-1, 1))

        n_iter = 6
        xlimits = np.array([[0.0, 25.0]])
        xdoe = np.atleast_2d([0, 7, 25]).T
        n_doe = xdoe.size

        criterion = "EI"  #'EI' or 'SBO' or 'UCB'

        ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits)

        x_opt, y_opt, ind_best, x_data, y_data, x_doe, y_doe = ego.optimize(
            fun=function_test_1d
        )
        print("Minimum in x={:.1f} with f(x)={:.1f}".format(float(x_opt), float(y_opt)))

        x_plot = np.atleast_2d(np.linspace(0, 25, 100)).T
        y_plot = function_test_1d(x_plot)

        fig = plt.figure(figsize=[10, 10])
        for i in range(n_iter):
            k = n_doe + i
            x_data_k = x_data[0:k]
            y_data_k = y_data[0:k]
            ego.gpr.set_training_values(x_data_k, y_data_k)
            ego.gpr.train()

            y_gp_plot = ego.gpr.predict_values(x_plot)
            y_gp_plot_var = ego.gpr.predict_variances(x_plot)
            y_ei_plot = -ego.EI(x_plot, y_data_k)

            ax = fig.add_subplot((n_iter + 1) // 2, 2, i + 1)
            ax1 = ax.twinx()
            ei, = ax1.plot(x_plot, y_ei_plot, color="red")

            true_fun, = ax.plot(x_plot, y_plot)
            data, = ax.plot(
                x_data_k, y_data_k, linestyle="", marker="o", color="orange"
            )
            if i < n_iter - 1:
                opt, = ax.plot(
                    x_data[k], y_data[k], linestyle="", marker="*", color="r"
                )
            gp, = ax.plot(x_plot, y_gp_plot, linestyle="--", color="g")
            sig_plus = y_gp_plot + 3 * y_gp_plot_var
            sig_moins = y_gp_plot - 3 * y_gp_plot_var
            un_gp = ax.fill_between(
                x_plot.T[0], sig_plus.T[0], sig_moins.T[0], alpha=0.3, color="g"
            )
            lines = [true_fun, data, gp, un_gp, opt, ei]
            fig.suptitle("EGO optimization of $f(x) = x \sin{x}$")
            fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.8)
            ax.set_title("iteration {}".format(i + 1))
            fig.legend(
                lines,
                [
                    "f(x)=xsin(x)",
                    "Given data points",
                    "Kriging prediction",
                    "Kriging 99% confidence interval",
                    "Next point to evaluate",
                    "Expected improvment function",
                ],
            )
        plt.show()
        # Check the optimal point is x_opt=18.9, y_opt =-15.1

    @staticmethod
    def run_ego_parallel_example():
        import numpy as np
        import six
        from smt.applications import EGO
        from smt.sampling_methods import FullFactorial

        import sklearn
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.stats import norm

        def function_test_1d(x):
            # function xsinx
            import numpy as np

            x = np.reshape(x, (-1,))
            y = np.zeros(x.shape)
            y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
            return y.reshape((-1, 1))

        n_iter = 3
        n_par = 3
        n_start = 50
        xlimits = np.array([[0.0, 25.0]])
        xdoe = np.atleast_2d([0, 7, 25]).T
        n_doe = xdoe.size

        criterion = "EI"  #'EI' or 'SBO' or 'UCB'
        qEIAproxCrit = "KBUB"  # "KB", "KBLB", "KBUB", "KBRand"
        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            xlimits=xlimits,
            n_par=n_par,
            qEIAproxCrit=qEIAproxCrit,
            n_start=n_start,
        )

        x_opt, y_opt, ind_best, x_data, y_data, x_doe, y_doe = ego.optimize(
            fun=function_test_1d
        )
        print("Minimum in x={:.1f} with f(x)={:.1f}".format(float(x_opt), float(y_opt)))

        x_plot = np.atleast_2d(np.linspace(0, 25, 100)).T
        y_plot = function_test_1d(x_plot)

        fig = plt.figure(figsize=[10, 10])
        for i in range(n_iter):
            k = n_doe + (i) * (n_par)
            x_data_k = x_data[0:k]
            y_data_k = y_data[0:k]
            x_data_sub = x_data_k.copy()
            y_data_sub = y_data_k.copy()
            for p in range(n_par):
                ego.gpr.set_training_values(x_data_sub, y_data_sub)
                ego.gpr.train()

                y_ei_plot = -ego.EI(x_plot, y_data_sub)
                y_gp_plot = ego.gpr.predict_values(x_plot)
                y_gp_plot_var = ego.gpr.predict_variances(x_plot)

                x_data_sub = np.append(x_data_sub, x_data[k + p])
                y_KB = ego.set_virtual_point(np.atleast_2d(x_data[k + p]), y_data_sub)

                y_data_sub = np.append(y_data_sub, y_KB)

                ax = fig.add_subplot(n_iter, n_par, i * (n_par) + p + 1)
                ax1 = ax.twinx()
                ei, = ax1.plot(x_plot, y_ei_plot, color="red")

                true_fun, = ax.plot(x_plot, y_plot)
                data, = ax.plot(
                    x_data_sub[: -1 - p],
                    y_data_sub[: -1 - p],
                    linestyle="",
                    marker="o",
                    color="orange",
                )
                virt_data, = ax.plot(
                    x_data_sub[-p - 1 : -1],
                    y_data_sub[-p - 1 : -1],
                    linestyle="",
                    marker="o",
                    color="g",
                )

                opt, = ax.plot(
                    x_data_sub[-1], y_data_sub[-1], linestyle="", marker="*", color="r"
                )
                gp, = ax.plot(x_plot, y_gp_plot, linestyle="--", color="g")
                sig_plus = y_gp_plot + 3.0 * np.sqrt(y_gp_plot_var)
                sig_moins = y_gp_plot - 3.0 * np.sqrt(y_gp_plot_var)
                un_gp = ax.fill_between(
                    x_plot.T[0], sig_plus.T[0], sig_moins.T[0], alpha=0.3, color="g"
                )
                lines = [true_fun, data, gp, un_gp, opt, ei, virt_data]
                fig.suptitle("EGOp optimization of $f(x) = x \sin{x}$")
                fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.8)
                ax.set_title("iteration {}.{}".format(i, p))
                fig.legend(
                    lines,
                    [
                        "f(x)=xsin(x)",
                        "Given data points",
                        "Kriging prediction",
                        "Kriging 99% confidence interval",
                        "Next point to evaluate",
                        "Expected improvment function",
                        "Virtula data points",
                    ],
                )
        plt.show()


if __name__ == "__main__":
    if "--plot" in argv:
        TestEGO.plot = True
        argv.remove("--plot")
    if "--example" in argv:
        TestEGO.run_ego_example()
        exit()
    unittest.main()
