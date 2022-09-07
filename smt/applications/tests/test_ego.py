# coding: utf-8
"""
Author: Remi Lafage <remi.lafage@onera.fr> and Nathalie Bartoli
This package is distributed under New BSD license.
"""

import warnings

warnings.filterwarnings("ignore")

import time
import sys
import unittest
import numpy as np
from sys import argv
import matplotlib

matplotlib.use("Agg")

from smt.applications import EGO
from smt.applications.ego import Evaluator
from smt.utils.sm_test_case import SMTestCase
from smt.problems import Branin, Rosenbrock
from smt.sampling_methods import FullFactorial
from multiprocessing import Pool
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG, GEKPLS, KPLS
from smt.surrogate_models import (
    FLOAT,
    ENUM,
    ORD,
    GOWER_KERNEL,
    EXP_HOMO_HSPHERE_KERNEL,
)
from smt.applications.mixed_integer import (
    MixedIntegerContext,
    MixedIntegerSamplingMethod,
)

# This implementation only works with Python > 3.3
class ParallelEvaluator(Evaluator):
    def run(self, fun, x):
        with Pool(3) as p:
            return np.array(
                [y[0] for y in p.map(fun, [np.atleast_2d(x[i]) for i in range(len(x))])]
            )


class TestEGO(SMTestCase):

    plot = None

    @staticmethod
    def function_test_1d(x):
        # function xsinx
        x = np.reshape(x, (-1,))
        y = np.zeros(x.shape)
        y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
        return y.reshape((-1, 1))

    def test_evaluator(self):
        x = [[1], [2], [3]]
        expected = TestEGO.function_test_1d(x)
        actual = ParallelEvaluator().run(TestEGO.function_test_1d, x)
        for i in range(len(x)):
            self.assertAlmostEqual(expected[i, 0], actual[i, 0])

    def test_function_test_1d(self):
        n_iter = 15
        xlimits = np.array([[0.0, 25.0]])

        criterion = "EI"

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            n_doe=3,
            xlimits=xlimits,
            random_state=42,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=TestEGO.function_test_1d)

        self.assertAlmostEqual(18.9, float(x_opt), delta=1)
        self.assertAlmostEqual(-15.1, float(y_opt), delta=1)

    def test_function_test_1d_parallel(self):
        n_iter = 3
        xlimits = np.array([[0.0, 25.0]])

        criterion = "EI"
        n_parallel = 3

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            n_doe=3,
            xlimits=xlimits,
            n_parallel=n_parallel,
            evaluator=ParallelEvaluator(),
            random_state=42,
        )
        x_opt, y_opt, _, _, _ = ego.optimize(fun=TestEGO.function_test_1d)

        self.assertAlmostEqual(18.9, float(x_opt), delta=1)
        self.assertAlmostEqual(-15.1, float(y_opt), delta=1)

    def test_rosenbrock_2D(self):
        n_iter = 50
        fun = Rosenbrock(ndim=2)
        xlimits = fun.xlimits
        criterion = "LCB"  #'EI' or 'SBO' or 'LCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        ego = EGO(
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            xlimits=xlimits,
            random_state=0,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)
        self.assertTrue(np.allclose([[1, 1]], x_opt, rtol=0.5))
        self.assertAlmostEqual(0.0, float(y_opt), delta=1)

    def test_rosenbrock_2D_parallel(self):
        n_iter = 20
        n_parallel = 5
        fun = Rosenbrock(ndim=2)
        xlimits = fun.xlimits
        criterion = "LCB"  #'EI' or 'SBO' or 'LCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        qEI = "KB"
        ego = EGO(
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            xlimits=xlimits,
            n_parallel=n_parallel,
            qEI=qEI,
            evaluator=ParallelEvaluator(),
            random_state=42,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)
        print("Rosenbrock: ", x_opt)
        self.assertTrue(np.allclose([[1, 1]], x_opt, rtol=0.5))
        self.assertAlmostEqual(0.0, float(y_opt), delta=1)

    def test_branin_2D(self):
        n_iter = 20
        fun = Branin(ndim=2)
        xlimits = fun.xlimits
        criterion = "LCB"  #'EI' or 'SBO' or 'LCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        ego = EGO(
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            xlimits=xlimits,
            random_state=42,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)
        # 3 optimal points possible: [-pi, 12.275], [pi, 2.275], [9.42478, 2.475]
        self.assertTrue(
            np.allclose([[-3.14, 12.275]], x_opt, rtol=0.2)
            or np.allclose([[3.14, 2.275]], x_opt, rtol=0.2)
            or np.allclose([[9.42, 2.475]], x_opt, rtol=0.2)
        )
        self.assertAlmostEqual(0.39, float(y_opt), delta=1)

    def test_branin_2D_parallel(self):
        n_iter = 10
        fun = Branin(ndim=2)
        n_parallel = 5
        xlimits = fun.xlimits
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        ego = EGO(
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            xlimits=xlimits,
            n_parallel=n_parallel,
            random_state=42,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)

        # 3 optimal points possible: [-pi, 12.275], [pi, 2.275], [9.42478, 2.475]
        self.assertTrue(
            np.allclose([[-3.14, 12.275]], x_opt, rtol=0.5)
            or np.allclose([[3.14, 2.275]], x_opt, rtol=0.5)
            or np.allclose([[9.42, 2.475]], x_opt, rtol=0.5)
        )
        print("Branin=", x_opt)
        self.assertAlmostEqual(0.39, float(y_opt), delta=1)

    def test_branin_2D_mixed_parallel(self):
        n_parallel = 5
        n_iter = 20
        fun = Branin(ndim=2)
        xlimits = fun.xlimits
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'
        qEI = "KB"
        xtypes = [ORD, FLOAT]

        sm = KRG(print_global=False)
        mixint = MixedIntegerContext(xtypes, xlimits)
        sampling = mixint.build_sampling_method(FullFactorial)
        xdoe = sampling(10)

        ego = EGO(
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            xtypes=[ORD, FLOAT],
            xlimits=xlimits,
            n_parallel=n_parallel,
            qEI=qEI,
            evaluator=ParallelEvaluator(),
            surrogate=sm,
            random_state=42,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)
        # 3 optimal points possible: [-pi, 12.275], [pi, 2.275], [9.42478, 2.475]
        self.assertTrue(
            np.allclose([[-3, 12.275]], x_opt, rtol=0.2)
            or np.allclose([[3, 2.275]], x_opt, rtol=0.2)
            or np.allclose([[9, 2.475]], x_opt, rtol=0.2)
        )
        self.assertAlmostEqual(0.494, float(y_opt), delta=1)

    def test_branin_2D_mixed(self):
        n_iter = 20
        fun = Branin(ndim=2)
        xtypes = [ORD, FLOAT]
        xlimits = fun.xlimits
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'

        sm = KRG(print_global=False)
        mixint = MixedIntegerContext(xtypes, xlimits)
        sampling = MixedIntegerSamplingMethod(xtypes, xlimits, FullFactorial)
        xdoe = sampling(10)

        ego = EGO(
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            xtypes=xtypes,
            xlimits=xlimits,
            surrogate=sm,
            random_state=42,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)
        # 3 optimal points possible: [-pi, 12.275], [pi, 2.275], [9.42478, 2.475]
        self.assertTrue(
            np.allclose([[-3, 12.275]], x_opt, rtol=0.2)
            or np.allclose([[3, 2.275]], x_opt, rtol=0.2)
            or np.allclose([[9, 2.475]], x_opt, rtol=0.2)
        )
        self.assertAlmostEqual(0.494, float(y_opt), delta=1)

    @staticmethod
    def function_test_mixed_integer(X):
        import numpy as np

        # float
        x1 = X[:, 0]
        #  enum 1
        c1 = X[:, 1]
        x2 = c1 == 0
        x3 = c1 == 1
        x4 = c1 == 2
        #  enum 2
        c2 = X[:, 2]
        x5 = c2 == 0
        x6 = c2 == 1
        # int
        i = X[:, 3]

        y = (
            (x2 + 2 * x3 + 3 * x4) * x5 * x1
            + (x2 + 2 * x3 + 3 * x4) * x6 * 0.95 * x1
            + i
        )
        return y

    def test_ego_mixed_integer(self):
        n_iter = 15
        xtypes = [FLOAT, (ENUM, 3), (ENUM, 2), ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red", "green"], ["large", "small"], ["0", "2", "3"]]
        )
        n_doe = 2
        sampling = MixedIntegerSamplingMethod(
            xtypes, xlimits, LHS, criterion="ese", random_state=42
        )
        xdoe = sampling(n_doe)
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'
        sm = KRG(print_global=False)
        mixint = MixedIntegerContext(xtypes, xlimits)

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            xtypes=xtypes,
            xlimits=xlimits,
            surrogate=sm,
            enable_tunneling=False,
            random_state=42,
        )
        _, y_opt, _, _, _ = ego.optimize(fun=TestEGO.function_test_mixed_integer)

        self.assertAlmostEqual(-15, float(y_opt), delta=5)

    def test_ego_mixed_integer_gower_distance(self):
        n_iter = 15
        xtypes = [FLOAT, (ENUM, 3), (ENUM, 2), ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red", "green"], ["large", "small"], [0, 2]]
        )
        n_doe = 2
        sampling = MixedIntegerSamplingMethod(
            xtypes,
            xlimits,
            LHS,
            criterion="ese",
            random_state=42,
            output_in_folded_space=True,
        )
        xdoe = sampling(n_doe)
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'
        sm = KRG(print_global=False)
        mixint = MixedIntegerContext(xtypes, xlimits)

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            xtypes=xtypes,
            xlimits=xlimits,
            surrogate=sm,
            enable_tunneling=False,
            random_state=42,
            categorical_kernel=GOWER_KERNEL,
        )
        _, y_opt, _, _, _ = ego.optimize(fun=TestEGO.function_test_mixed_integer)

        self.assertAlmostEqual(-15, float(y_opt), delta=5)

    def test_ego_mixed_integer_homo_gaussian(self):
        n_iter = 15
        xtypes = [FLOAT, (ENUM, 3), (ENUM, 2), ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red", "green"], ["large", "small"], [0, 2]]
        )
        n_doe = 2
        sampling = MixedIntegerSamplingMethod(
            xtypes,
            xlimits,
            LHS,
            criterion="ese",
            random_state=42,
            output_in_folded_space=True,
        )
        xdoe = sampling(n_doe)
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'
        sm = KRG(print_global=False)
        mixint = MixedIntegerContext(xtypes, xlimits)

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            xtypes=xtypes,
            xlimits=xlimits,
            surrogate=sm,
            enable_tunneling=False,
            random_state=42,
            categorical_kernel=EXP_HOMO_HSPHERE_KERNEL,
        )
        _, y_opt, _, _, _ = ego.optimize(fun=TestEGO.function_test_mixed_integer)

        self.assertAlmostEqual(-15, float(y_opt), delta=5)

    def test_ego_mixed_integer_homo_gaussian_pls(self):
        n_iter = 15
        xtypes = [FLOAT, (ENUM, 3), (ENUM, 2), ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red", "green"], ["large", "small"], [0, 2]]
        )
        n_doe = 7
        sampling = MixedIntegerSamplingMethod(
            xtypes,
            xlimits,
            LHS,
            criterion="ese",
            random_state=42,
            output_in_folded_space=True,
        )
        xdoe = sampling(n_doe)
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'
        sm = KPLS(print_global=False, n_comp=1, cat_kernel_comps=[2, 2])
        mixint = MixedIntegerContext(xtypes, xlimits)

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            xtypes=xtypes,
            xlimits=xlimits,
            surrogate=sm,
            enable_tunneling=False,
            random_state=42,
            categorical_kernel=EXP_HOMO_HSPHERE_KERNEL,
        )
        _, y_opt, _, _, _ = ego.optimize(fun=TestEGO.function_test_mixed_integer)

        self.assertAlmostEqual(-15, float(y_opt), delta=5)

    def test_ydoe_option(self):
        n_iter = 15
        fun = Branin(ndim=2)
        xlimits = fun.xlimits
        criterion = "LCB"  #'EI' or 'SBO' or 'LCB'

        xdoe = FullFactorial(xlimits=xlimits)(10)
        ydoe = fun(xdoe)

        ego = EGO(
            xdoe=xdoe,
            ydoe=ydoe,
            n_iter=n_iter,
            criterion=criterion,
            xlimits=xlimits,
            random_state=42,
        )
        _, y_opt, _, _, _ = ego.optimize(fun=fun)

        self.assertAlmostEqual(0.39, float(y_opt), delta=1)

    def test_find_best_point(self):
        fun = TestEGO.function_test_1d
        xlimits = np.array([[0.0, 25.0]])
        xdoe = FullFactorial(xlimits=xlimits)(3)
        ydoe = fun(xdoe)
        ego = EGO(
            xdoe=xdoe,
            ydoe=ydoe,
            n_iter=1,
            criterion="LCB",
            xlimits=xlimits,
            n_start=30,
            enable_tunneling=False,
            random_state=42,
        )
        _, _, _, _, _ = ego.optimize(fun=fun)
        x, _ = ego._find_best_point(xdoe, ydoe, enable_tunneling=False)
        self.assertAlmostEqual(6.5, float(x), delta=1)

    @staticmethod
    def initialize_ego_gek(func="exp", criterion="LCB"):
        from smt.problems import TensorProduct

        class TensorProductIndirect(TensorProduct):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.super = super()

            def _evaluate(self, x, kx):
                assert kx is None
                response = self.super._evaluate(x, kx)
                sens = np.hstack(
                    self.super._evaluate(x, ki) for ki in range(x.shape[1])
                )
                return np.hstack((response, sens))

        fun = TensorProductIndirect(ndim=2, func=func)

        # Construction of the DOE
        sampling = LHS(xlimits=fun.xlimits, criterion="m", random_state=42)
        xdoe = sampling(20)
        ydoe = fun(xdoe)

        # Build the GEKPLS surrogate model
        n_comp = 2
        sm = GEKPLS(
            theta0=[1e-2] * n_comp,
            xlimits=fun.xlimits,
            extra_points=1,
            print_prediction=False,
            n_comp=n_comp,
        )

        # Build the EGO optimizer and optimize
        ego = EGO(
            xdoe=xdoe,
            ydoe=ydoe,
            n_iter=5,
            criterion=criterion,
            xlimits=fun.xlimits,
            surrogate=sm,
            n_start=30,
            enable_tunneling=False,
            random_state=42,
        )

        return ego, fun

    def test_ego_gek(self):
        ego, fun = self.initialize_ego_gek()
        x_opt, _, _, _, _ = ego.optimize(fun=fun)

        self.assertAlmostEqual(-1.0, float(x_opt[0]), delta=1e-4)
        self.assertAlmostEqual(-1.0, float(x_opt[1]), delta=1e-4)

    def test_ei_gek(self):
        ego, fun = self.initialize_ego_gek(func="cos", criterion="EI")
        x_data, y_data = ego._setup_optimizer(fun)
        ego._train_gpr(x_data, y_data)

        # Test the EI value at the following point
        ei = ego.EI(np.array([[0.8398599985874058, -0.3240337426231973]]))

        self.assertTrue(np.allclose(ei, [6.87642e-12, 1.47804e-10, 2.76223], atol=1e-1))

    def test_qei_criterion_default(self):
        fun = TestEGO.function_test_1d
        xlimits = np.array([[0.0, 25.0]])
        xdoe = FullFactorial(xlimits=xlimits)(3)
        ydoe = fun(xdoe)
        ego = EGO(
            xdoe=xdoe, ydoe=ydoe, n_iter=1, criterion="SBO", xlimits=xlimits, n_start=30
        )
        ego._setup_optimizer(fun)
        ego.gpr.set_training_values(xdoe, ydoe)
        ego.gpr.train()
        xtest = np.array([[10.0]])
        # test that default virtual point should be equal to 3sigma lower bound kriging interval
        expected = float(
            ego.gpr.predict_values(xtest)
            - 3 * np.sqrt(ego.gpr.predict_variances(xtest))
        )
        actual = float(ego._get_virtual_point(xtest, fun(xtest))[0])
        self.assertAlmostEqual(expected, actual)

    @staticmethod
    def run_ego_example():
        import numpy as np
        from smt.applications import EGO
        import matplotlib.pyplot as plt

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

        criterion = "EI"  #'EI' or 'SBO' or 'LCB'

        ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits)

        x_opt, y_opt, _, x_data, y_data = ego.optimize(fun=function_test_1d)
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
            y_ei_plot = -ego.EI(x_plot)

            ax = fig.add_subplot((n_iter + 1) // 2, 2, i + 1)
            ax1 = ax.twinx()
            (ei,) = ax1.plot(x_plot, y_ei_plot, color="red")

            (true_fun,) = ax.plot(x_plot, y_plot)
            (data,) = ax.plot(
                x_data_k, y_data_k, linestyle="", marker="o", color="orange"
            )
            if i < n_iter - 1:
                (opt,) = ax.plot(
                    x_data[k], y_data[k], linestyle="", marker="*", color="r"
                )
            (gp,) = ax.plot(x_plot, y_gp_plot, linestyle="--", color="g")
            sig_plus = y_gp_plot + 3 * np.sqrt(y_gp_plot_var)
            sig_moins = y_gp_plot - 3 * np.sqrt(y_gp_plot_var)
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
    def run_ego_mixed_integer_example():
        import numpy as np
        from smt.applications import EGO
        from smt.applications.mixed_integer import (
            MixedIntegerContext,
            FLOAT,
            ENUM,
            ORD,
        )
        import matplotlib.pyplot as plt
        from smt.surrogate_models import KRG
        from smt.sampling_methods import LHS

        # Regarding the interface, the function to be optimized should handle
        # categorical values as index values in the enumeration type specification.
        # For instance, here "blue" will be passed to the function as the index value 2.
        # This allows to keep the numpy ndarray X handling numerical values.
        def function_test_mixed_integer(X):
            # float
            x1 = X[:, 0]
            #  enum 1
            c1 = X[:, 1]
            x2 = c1 == 0
            x3 = c1 == 1
            x4 = c1 == 2
            #  enum 2
            c2 = X[:, 2]
            x5 = c2 == 0
            x6 = c2 == 1
            # int
            i = X[:, 3]

            y = (
                (x2 + 2 * x3 + 3 * x4) * x5 * x1
                + (x2 + 2 * x3 + 3 * x4) * x6 * 0.95 * x1
                + i
            )
            return y

        n_iter = 15
        xtypes = [FLOAT, (ENUM, 3), (ENUM, 2), ORD]
        xlimits = np.array(
            [[-5, 5], ["red", "green", "blue"], ["square", "circle"], [0, 2]]
        )
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'
        qEI = "KB"
        sm = KRG(print_global=False)
        mixint = MixedIntegerContext(xtypes, xlimits)
        n_doe = 3
        sampling = mixint.build_sampling_method(LHS, criterion="ese", random_state=42)
        xdoe = sampling(n_doe)
        ydoe = function_test_mixed_integer(xdoe)

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            ydoe=ydoe,
            xtypes=xtypes,
            xlimits=xlimits,
            surrogate=sm,
            qEI=qEI,
            random_state=42,
        )

        x_opt, y_opt, _, _, y_data = ego.optimize(fun=function_test_mixed_integer)
        print("Minimum in x={} with f(x)={:.1f}".format(x_opt, float(y_opt)))
        print("Minimum in typed x={}".format(ego.mixint.cast_to_mixed_integer(x_opt)))

        min_ref = -15
        mini = np.zeros(n_iter)
        for k in range(n_iter):
            mini[k] = np.log(np.abs(np.min(y_data[0 : k + n_doe - 1]) - min_ref))
        x_plot = np.linspace(1, n_iter + 0.5, n_iter)
        u = max(np.floor(max(mini)) + 1, -100)
        l = max(np.floor(min(mini)) - 0.2, -10)
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(x_plot, mini, color="r")
        axes.set_ylim([l, u])
        plt.title("minimum convergence plot", loc="center")
        plt.xlabel("number of iterations")
        plt.ylabel("log of the difference w.r.t the best")
        plt.show()

    @staticmethod
    def run_ego_parallel_example():
        import numpy as np
        from smt.applications import EGO
        from smt.applications.ego import EGO, Evaluator
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
        n_parallel = 3
        n_start = 50
        xlimits = np.array([[0.0, 25.0]])
        xdoe = np.atleast_2d([0, 7, 25]).T
        n_doe = xdoe.size

        class ParallelEvaluator(Evaluator):
            """
            Implement Evaluator interface using multiprocessing ThreadPool object (Python 3 only).
            """

            def run(self, fun, x):
                n_thread = 5
                # Caveat: import are made here due to SMT documentation building process
                import numpy as np
                from sys import version_info
                from multiprocessing.pool import ThreadPool

                if version_info.major == 2:
                    return fun(x)
                # Python 3 only
                with ThreadPool(n_thread) as p:
                    return np.array(
                        [
                            y[0]
                            for y in p.map(
                                fun, [np.atleast_2d(x[i]) for i in range(len(x))]
                            )
                        ]
                    )

        criterion = "EI"  #'EI' or 'SBO' or 'LCB'
        qEI = "KBUB"  # "KB", "KBLB", "KBUB", "KBRand"
        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            xlimits=xlimits,
            n_parallel=n_parallel,
            qEI=qEI,
            n_start=n_start,
            evaluator=ParallelEvaluator(),
            random_state=42,
        )

        x_opt, y_opt, _, x_data, y_data = ego.optimize(fun=function_test_1d)
        print("Minimum in x={:.1f} with f(x)={:.1f}".format(float(x_opt), float(y_opt)))

        x_plot = np.atleast_2d(np.linspace(0, 25, 100)).T
        y_plot = function_test_1d(x_plot)

        fig = plt.figure(figsize=[10, 10])
        for i in range(n_iter):
            k = n_doe + (i) * (n_parallel)
            x_data_k = x_data[0:k]
            y_data_k = y_data[0:k]
            x_data_sub = x_data_k.copy()
            y_data_sub = y_data_k.copy()
            for p in range(n_parallel):
                ego.gpr.set_training_values(x_data_sub, y_data_sub)
                ego.gpr.train()

                y_ei_plot = -ego.EI(x_plot)
                y_gp_plot = ego.gpr.predict_values(x_plot)
                y_gp_plot_var = ego.gpr.predict_variances(x_plot)

                x_data_sub = np.append(x_data_sub, x_data[k + p])
                y_KB = ego._get_virtual_point(np.atleast_2d(x_data[k + p]), y_data_sub)

                y_data_sub = np.append(y_data_sub, y_KB)

                ax = fig.add_subplot(n_iter, n_parallel, i * (n_parallel) + p + 1)
                ax1 = ax.twinx()
                (ei,) = ax1.plot(x_plot, y_ei_plot, color="red")

                (true_fun,) = ax.plot(x_plot, y_plot)
                (data,) = ax.plot(
                    x_data_sub[: -1 - p],
                    y_data_sub[: -1 - p],
                    linestyle="",
                    marker="o",
                    color="orange",
                )
                (virt_data,) = ax.plot(
                    x_data_sub[-p - 1 : -1],
                    y_data_sub[-p - 1 : -1],
                    linestyle="",
                    marker="o",
                    color="g",
                )

                (opt,) = ax.plot(
                    x_data_sub[-1], y_data_sub[-1], linestyle="", marker="*", color="r"
                )
                (gp,) = ax.plot(x_plot, y_gp_plot, linestyle="--", color="g")
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
        TestEGO.run_ego_mixed_integer_example()
        exit()
    unittest.main()
