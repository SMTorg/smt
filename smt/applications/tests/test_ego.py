# coding: utf-8
"""
Author: Remi Lafage <remi.lafage@onera.fr> and Nathalie Bartoli
This package is distributed under New BSD license.
"""

import warnings

warnings.filterwarnings("ignore")

import os
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
from smt.surrogate_models import (
    KRG,
    GEKPLS,
    KPLS,
    MixIntKernelType,
    DesignSpace,
    OrdinalVariable,
    FloatVariable,
    CategoricalVariable,
    IntegerVariable,
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
        design_space = DesignSpace(xlimits)

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            n_doe=3,
            surrogate=KRG(design_space=design_space, print_global=False),
            random_state=42,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=TestEGO.function_test_1d)

        self.assertAlmostEqual(18.9, float(x_opt), delta=1)
        self.assertAlmostEqual(-15.1, float(y_opt), delta=1)

    def test_function_test_1d_parallel(self):
        n_iter = 3
        xlimits = np.array([[0.0, 25.0]])
        design_space = DesignSpace(xlimits)

        criterion = "EI"
        n_parallel = 3

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            n_doe=3,
            surrogate=KRG(design_space=design_space, print_global=False),
            n_parallel=n_parallel,
            evaluator=ParallelEvaluator(),
            random_state=42,
        )
        x_opt, y_opt, _, _, _ = ego.optimize(fun=TestEGO.function_test_1d)

        self.assertAlmostEqual(18.9, float(x_opt), delta=1)
        self.assertAlmostEqual(-15.1, float(y_opt), delta=1)

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_rosenbrock_2D(self):
        n_iter = 50
        fun = Rosenbrock(ndim=2)
        xlimits = fun.xlimits
        criterion = "LCB"  #'EI' or 'SBO' or 'LCB'
        random_state = 42
        design_space = DesignSpace(xlimits, seed=random_state)
        xdoe = FullFactorial(xlimits=xlimits)(10)
        ego = EGO(
            n_start=30,
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            surrogate=KRG(design_space=design_space, n_start=25, print_global=False),
            random_state=random_state,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)
        self.assertTrue(np.allclose([[1, 1]], x_opt, rtol=0.55))
        self.assertAlmostEqual(0.0, float(y_opt), delta=1)

    def test_rosenbrock_2D_SBO(self):
        n_iter = 10
        fun = Rosenbrock(ndim=2)
        xlimits = fun.xlimits
        criterion = "SBO"  #'EI' or 'SBO' or 'LCB'
        design_space = DesignSpace(xlimits)

        xdoe = FullFactorial(xlimits=xlimits)(50)
        ego = EGO(
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            surrogate=KRG(design_space=design_space, print_global=False),
            random_state=42,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)
        self.assertTrue(np.allclose([[1, 1]], x_opt, atol=1))
        self.assertAlmostEqual(0.0, float(y_opt), delta=1)

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_rosenbrock_2D_parallel(self):
        n_iter = 20
        n_parallel = 5
        fun = Rosenbrock(ndim=2)
        xlimits = fun.xlimits
        criterion = "LCB"  #'EI' or 'SBO' or 'LCB'
        random_state = 42
        design_space = DesignSpace(xlimits, seed=random_state)

        xdoe = FullFactorial(xlimits=xlimits)(10)
        qEI = "KB"
        ego = EGO(
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            surrogate=KRG(design_space=design_space, print_global=False),
            n_parallel=n_parallel,
            qEI=qEI,
            evaluator=ParallelEvaluator(),
            random_state=random_state,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)
        print("Rosenbrock: ", x_opt)
        self.assertTrue(np.allclose([[1, 1]], x_opt, rtol=0.5))
        self.assertAlmostEqual(0.0, float(y_opt), delta=1)

    def test_branin_2D(self):
        n_iter = 20
        fun = Branin(ndim=2)
        criterion = "LCB"  #'EI' or 'SBO' or 'LCB'
        design_space = fun.design_space
        ego = EGO(
            surrogate=KRG(design_space=design_space, print_global=False),
            n_iter=n_iter,
            criterion=criterion,
            n_doe=10,
            random_state=42,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)
        # 3 optimal points possible: [-pi, 12.275], [pi, 2.275], [9.42478, 2.475]
        self.assertTrue(
            np.allclose([[-3.14, 12.275]], x_opt, rtol=0.25)
            or np.allclose([[3.14, 2.275]], x_opt, rtol=0.25)
            or np.allclose([[9.42, 2.475]], x_opt, rtol=0.25)
        )
        self.assertAlmostEqual(0.39, float(y_opt), delta=0.8)

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_branin_2D_parallel(self):
        n_iter = 10
        fun = Branin(ndim=2)
        n_parallel = 5
        xlimits = fun.xlimits
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'
        design_space = DesignSpace(xlimits)

        xdoe = FullFactorial(xlimits=xlimits)(10)
        ego = EGO(
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            surrogate=KRG(design_space=design_space, print_global=False),
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

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_branin_2D_mixed_parallel(self):
        n_parallel = 5
        n_iter = 20
        fun = Branin(ndim=2)
        xlimits = fun.xlimits
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'
        qEI = "CLmin"
        random_state = 42
        design_space = DesignSpace(
            [
                IntegerVariable(*xlimits[0]),
                FloatVariable(*xlimits[1]),
            ],
            seed=random_state,
        )
        sm = KRG(design_space=design_space, print_global=False, n_start=25)
        mixint = MixedIntegerContext(design_space)
        sampling = mixint.build_sampling_method(FullFactorial)
        xdoe = sampling(10)

        ego = EGO(
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            n_parallel=n_parallel,
            qEI=qEI,
            n_start=30,
            evaluator=ParallelEvaluator(),
            surrogate=sm,
            random_state=random_state,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)
        # 3 optimal points possible: [-pi, 12.275], [pi, 2.275], [9.42478, 2.475]
        self.assertTrue(
            np.allclose([[-3, 12.275]], x_opt, rtol=0.2)
            or np.allclose([[3, 2.275]], x_opt, rtol=0.2)
            or np.allclose([[9, 2.475]], x_opt, rtol=0.2)
        )
        self.assertAlmostEqual(0.494, float(y_opt), delta=1)

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_branin_2D_mixed(self):
        n_iter = 20
        fun = Branin(ndim=2)
        xlimits = fun.xlimits
        random_state = 42
        design_space = DesignSpace(
            [
                IntegerVariable(*xlimits[0]),
                FloatVariable(*xlimits[1]),
            ],
            seed=random_state,
        )
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'

        sm = KRG(design_space=design_space, print_global=False)
        sampling = MixedIntegerSamplingMethod(FullFactorial, design_space)
        xdoe = sampling(10)

        ego = EGO(
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            surrogate=sm,
            enable_tunneling=False,
            random_state=random_state,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)
        # 3 optimal points possible: [-pi, 12.275], [pi, 2.275], [9.42478, 2.475]
        self.assertTrue(
            np.allclose([[-3, 12.275]], x_opt, rtol=0.2)
            or np.allclose([[3, 2.275]], x_opt, rtol=0.2)
            or np.allclose([[9, 2.475]], x_opt, rtol=0.2)
        )
        self.assertAlmostEqual(0.494, float(y_opt), delta=1)

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_branin_2D_mixed_tunnel(self):
        n_iter = 20
        fun = Branin(ndim=2)
        xlimits = fun.xlimits
        random_state = 42
        design_space = DesignSpace(
            [
                IntegerVariable(*xlimits[0]),
                FloatVariable(*xlimits[1]),
            ],
            seed=random_state,
        )
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'

        sm = KRG(design_space=design_space, print_global=False)
        sampling = MixedIntegerSamplingMethod(FullFactorial, design_space)
        xdoe = sampling(30)

        ego = EGO(
            xdoe=xdoe,
            n_iter=n_iter,
            criterion=criterion,
            surrogate=sm,
            enable_tunneling=True,
            random_state=random_state,
        )

        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)
        # 3 optimal points possible: [-pi, 12.275], [pi, 2.275], [9.42478, 2.475]
        self.assertTrue(
            np.allclose([[-3, 12.275]], x_opt, rtol=2)
            or np.allclose([[3, 2.275]], x_opt, rtol=2)
            or np.allclose([[9, 2.475]], x_opt, rtol=2)
        )
        self.assertAlmostEqual(0.494, float(y_opt), delta=2)

    @staticmethod
    def function_test_mixed_integer(X):
        import numpy as np

        # float
        x1 = X[:, 0]
        #  XType.ENUM 1
        c1 = X[:, 1]
        x2 = c1 == 0
        x3 = c1 == 1
        x4 = c1 == 2
        #  XType.ENUM 2
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

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_ego_mixed_integer(self):
        n_iter = 15
        n_doe = 5
        random_state = 42
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["blue", "red", "green"]),
                CategoricalVariable(["large", "small"]),
                OrdinalVariable([0, 2, 3]),
            ],
            seed=random_state,
        )
        xdoe, _ = design_space.sample_valid_x(n_doe)

        criterion = "EI"  #'EI' or 'SBO' or 'LCB'
        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            surrogate=KRG(design_space=design_space, print_global=False),
            enable_tunneling=False,
            random_state=random_state,
        )
        _, y_opt, _, _, _ = ego.optimize(fun=TestEGO.function_test_mixed_integer)

        self.assertAlmostEqual(-15, float(y_opt), delta=5)

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_ego_mixed_integer_gower_distance(self):
        n_iter = 15
        n_doe = 5
        random_state = 42
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["blue", "red", "green"]),
                CategoricalVariable(["large", "small"]),
                IntegerVariable(0, 2),
            ],
            seed=random_state,
        )
        xdoe, _ = design_space.sample_valid_x(n_doe)

        criterion = "EI"  #'EI' or 'SBO' or 'LCB'
        ego = EGO(
            n_start=30,
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            surrogate=KRG(
                n_start=25,
                design_space=design_space,
                categorical_kernel=MixIntKernelType.GOWER,
                print_global=False,
            ),
            enable_tunneling=False,
            random_state=random_state,
        )
        _, y_opt, _, _, _ = ego.optimize(fun=TestEGO.function_test_mixed_integer)

        self.assertAlmostEqual(-15, float(y_opt), delta=5)

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_ego_mixed_integer_hierarchical_NN(self):
        random_state = 42

        def f_neu(x1, x2, x3, x4):
            if x4 == 0:
                return 2 * x1 + x2 - 0.5 * x3
            elif x4 == 1:
                return -x1 + 2 * x2 - 0.5 * x3
            elif x4 == 2:
                return -x1 + x2 + 0.5 * x3
            else:
                raise ValueError(f"Unexpected x4: {x4}")

        def f1(x1, x2, x3, x4, x5):
            return f_neu(x1, x2, x3, x4) + x5**2

        def f2(x1, x2, x3, x4, x5, x6):
            return f_neu(x1, x2, x3, x4) + (x5**2) + 0.3 * x6

        def f3(x1, x2, x3, x4, x5, x6, x7):
            return f_neu(x1, x2, x3, x4) + (x5**2) + 0.3 * x6 - 0.1 * x7**3

        def f_hv(X):
            y = []
            for x in X:
                x3_decoded = design_space.decode_values(x, i_dv=3)[0]
                if x[0] == 0:
                    y.append(f1(x[1], x[2], x3_decoded, x[4], x[5]))
                elif x[0] == 1:
                    y.append(f2(x[1], x[2], x3_decoded, x[4], x[5], x[6]))
                elif x[0] == 2:
                    y.append(f3(x[1], x[2], x3_decoded, x[4], x[5], x[6], x[7]))
                else:
                    raise ValueError(f"Unexpected x0: {x[0]}")
            return np.array(y)

        random_state = 42
        design_space = DesignSpace(
            [
                OrdinalVariable(values=[1, 2, 3]),  # x0
                FloatVariable(-5, 2),
                FloatVariable(-5, 2),
                OrdinalVariable(values=[8, 16, 32, 64, 128, 256]),  # x3
                CategoricalVariable(values=["ReLU", "SELU", "ISRLU"]),  # x4
                IntegerVariable(0, 5),  # x5
                IntegerVariable(0, 5),  # x6
                IntegerVariable(0, 5),  # x7
            ],
            seed=random_state,
        )

        # x6 is active when x0 >= 2
        design_space.declare_decreed_var(decreed_var=6, meta_var=0, meta_value=[2, 3])
        # x7 is active when x0 >= 3
        design_space.declare_decreed_var(decreed_var=7, meta_var=0, meta_value=3)

        n_doe = 4

        neutral_var_ds = DesignSpace(design_space.design_variables[1:])
        sampling = MixedIntegerSamplingMethod(
            LHS, neutral_var_ds, criterion="ese", random_state=random_state
        )
        x_cont = sampling(3 * n_doe)

        xdoe1 = np.zeros((n_doe, 8))
        x_cont2 = x_cont[:n_doe, :5]
        xdoe1[:, 0] = np.zeros(n_doe)
        xdoe1[:, 1:6] = x_cont2
        # ydoe1 = f_hv(xdoe1)

        xdoe1 = np.zeros((n_doe, 8))
        xdoe1[:, 0] = np.zeros(n_doe)
        xdoe1[:, 1:6] = x_cont2

        xdoe2 = np.zeros((n_doe, 8))
        x_cont2 = x_cont[n_doe : 2 * n_doe, :6]
        xdoe2[:, 0] = np.ones(n_doe)
        xdoe2[:, 1:7] = x_cont2
        # ydoe2 = f_hv(xdoe2)

        xdoe2 = np.zeros((n_doe, 8))
        xdoe2[:, 0] = np.ones(n_doe)
        xdoe2[:, 1:7] = x_cont2

        xdoe3 = np.zeros((n_doe, 8))
        xdoe3[:, 0] = 2 * np.ones(n_doe)
        xdoe3[:, 1:] = x_cont[2 * n_doe :, :]
        # ydoe3 = f_hv(xdoe3)

        Xt = np.concatenate((xdoe1, xdoe2, xdoe3), axis=0)
        # Yt = np.concatenate((ydoe1, ydoe2, ydoe3), axis=0)

        n_iter = 6
        criterion = "EI"

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=Xt,
            surrogate=KRG(
                design_space=design_space,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                theta0=[1e-2],
                n_start=5,
                corr="abs_exp",
                print_global=False,
            ),
            enable_tunneling=False,
            random_state=random_state,
        )

        x_opt, y_opt, dnk, x_data, y_data = ego.optimize(fun=f_hv)
        self.assertAlmostEqual(
            f_hv(np.atleast_2d([2, -5, -5, 5, 0, 0, 0, 5])),
            float(y_opt),
            delta=15,
        )

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_ego_mixed_integer_hierarchical_Goldstein(self):
        def H(x1, x2, x3, x4, z3, z4, x5, cos_term):
            h = (
                53.3108
                + 0.184901 * x1
                - 5.02914 * x1**3 * 10 ** (-6)
                + 7.72522 * x1**z3 * 10 ** (-8)
                - 0.0870775 * x2
                - 0.106959 * x3
                + 7.98772 * x3**z4 * 10 ** (-6)
                + 0.00242482 * x4
                + 1.32851 * x4**3 * 10 ** (-6)
                - 0.00146393 * x1 * x2
                - 0.00301588 * x1 * x3
                - 0.00272291 * x1 * x4
                + 0.0017004 * x2 * x3
                + 0.0038428 * x2 * x4
                - 0.000198969 * x3 * x4
                + 1.86025 * x1 * x2 * x3 * 10 ** (-5)
                - 1.88719 * x1 * x2 * x4 * 10 ** (-6)
                + 2.50923 * x1 * x3 * x4 * 10 ** (-5)
                - 5.62199 * x2 * x3 * x4 * 10 ** (-5)
            )
            if cos_term:
                h += 5.0 * np.cos(2.0 * np.pi * (x5 / 100.0)) - 2.0
            return h

        def f1(x1, x2, z1, z2, z3, z4, x5, cos_term):
            c1 = z2 == 0
            c2 = z2 == 1
            c3 = z2 == 2

            c4 = z3 == 0
            c5 = z3 == 1
            c6 = z3 == 2

            y = (
                c4
                * (
                    c1 * H(x1, x2, 20, 20, z3, z4, x5, cos_term)
                    + c2 * H(x1, x2, 50, 20, z3, z4, x5, cos_term)
                    + c3 * H(x1, x2, 80, 20, z3, z4, x5, cos_term)
                )
                + c5
                * (
                    c1 * H(x1, x2, 20, 50, z3, z4, x5, cos_term)
                    + c2 * H(x1, x2, 50, 50, z3, z4, x5, cos_term)
                    + c3 * H(x1, x2, 80, 50, z3, z4, x5, cos_term)
                )
                + c6
                * (
                    c1 * H(x1, x2, 20, 80, z3, z4, x5, cos_term)
                    + c2 * H(x1, x2, 50, 80, z3, z4, x5, cos_term)
                    + c3 * H(x1, x2, 80, 80, z3, z4, x5, cos_term)
                )
            )
            return y

        def f2(x1, x2, x3, z2, z3, z4, x5, cos_term):
            c1 = z2 == 0
            c2 = z2 == 1
            c3 = z2 == 2

            y = (
                c1 * H(x1, x2, x3, 20, z3, z4, x5, cos_term)
                + c2 * H(x1, x2, x3, 50, z3, z4, x5, cos_term)
                + c3 * H(x1, x2, x3, 80, z3, z4, x5, cos_term)
            )
            return y

        def f3(x1, x2, x4, z1, z3, z4, x5, cos_term):
            c1 = z1 == 0
            c2 = z1 == 1
            c3 = z1 == 2

            y = (
                c1 * H(x1, x2, 20, x4, z3, z4, x5, cos_term)
                + c2 * H(x1, x2, 50, x4, z3, z4, x5, cos_term)
                + c3 * H(x1, x2, 80, x4, z3, z4, x5, cos_term)
            )
            return y

        def f_hv(X):
            y = []
            for x in X:
                if x[0] == 0:
                    y.append(
                        f1(x[2], x[3], x[7], x[8], x[9], x[10], x[6], cos_term=x[1])
                    )
                elif x[0] == 1:
                    y.append(
                        f2(x[2], x[3], x[4], x[8], x[9], x[10], x[6], cos_term=x[1])
                    )
                elif x[0] == 2:
                    y.append(
                        f3(x[2], x[3], x[5], x[7], x[9], x[10], x[6], cos_term=x[1])
                    )
                elif x[0] == 3:
                    y.append(
                        H(x[2], x[3], x[4], x[5], x[9], x[10], x[6], cos_term=x[1])
                    )
                else:
                    raise ValueError
            return np.array(y)

        random_state = 0
        ds = DesignSpace(
            [
                CategoricalVariable(values=[0, 1, 2, 3]),  # meta
                OrdinalVariable(values=[0, 1]),  # x1
                FloatVariable(0, 100),  # x2
                FloatVariable(0, 100),
                FloatVariable(0, 100),
                FloatVariable(0, 100),
                FloatVariable(0, 100),
                IntegerVariable(0, 2),  # x7
                IntegerVariable(0, 2),
                IntegerVariable(0, 2),
                IntegerVariable(0, 2),
            ],
            seed=random_state,
        )

        # x4 is acting if meta == 1, 3
        ds.declare_decreed_var(decreed_var=4, meta_var=0, meta_value=[1, 3])
        # x5 is acting if meta == 2, 3
        ds.declare_decreed_var(decreed_var=5, meta_var=0, meta_value=[2, 3])
        # x7 is acting if meta == 0, 2
        ds.declare_decreed_var(decreed_var=7, meta_var=0, meta_value=[0, 2])
        # x8 is acting if meta == 0, 1
        ds.declare_decreed_var(decreed_var=8, meta_var=0, meta_value=[0, 1])

        n_doe = 25
        ds.seed = random_state
        Xt, x_is_active = ds.sample_valid_x(n_doe)

        n_iter = 10
        criterion = "EI"

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=Xt,
            surrogate=KRG(
                design_space=ds,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                theta0=[1e-2],
                n_start=10,
                corr="squar_exp",
                print_global=False,
            ),
            verbose=True,
            enable_tunneling=False,
            random_state=random_state,
            n_start=25,
        )

        x_opt, y_opt, dnk, x_data, y_data = ego.optimize(fun=f_hv)
        self.assertAlmostEqual(
            9.022,
            float(y_opt),
            delta=25,
        )

    def test_ego_mixed_integer_homo_gaussian(self):
        n_iter = 15
        random_state = 42
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["blue", "red", "green"]),
                CategoricalVariable(["large", "small"]),
                IntegerVariable(0, 2),
            ],
            seed=random_state,
        )
        n_doe = 5
        sampling = MixedIntegerSamplingMethod(
            LHS,
            design_space,
            criterion="ese",
            random_state=random_state,
            output_in_folded_space=True,
        )
        xdoe = sampling(n_doe)
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            surrogate=KRG(
                design_space=design_space,
                categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE,
                print_global=False,
            ),
            enable_tunneling=False,
            random_state=random_state,
        )
        _, y_opt, _, _, _ = ego.optimize(fun=TestEGO.function_test_mixed_integer)

        self.assertAlmostEqual(-15, float(y_opt), delta=5)

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_ego_mixed_integer_homo_gaussian_pls(self):
        n_iter = 15
        random_state = 42
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["blue", "red", "green"]),
                CategoricalVariable(["large", "small"]),
                IntegerVariable(0, 2),
            ],
            seed=random_state,
        )
        sampling = MixedIntegerSamplingMethod(
            LHS,
            design_space,
            criterion="ese",
            random_state=random_state,
            output_in_folded_space=True,
        )
        n_doe = 5
        xdoe = sampling(n_doe)
        criterion = "EI"  #'EI' or 'SBO' or 'LCB'
        sm = KPLS(
            print_global=False,
            design_space=design_space,
            categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE,
            n_comp=1,
            cat_kernel_comps=[2, 2],
        )
        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            surrogate=sm,
            enable_tunneling=False,
            random_state=random_state,
        )
        _, y_opt, _, _, _ = ego.optimize(fun=TestEGO.function_test_mixed_integer)

        self.assertAlmostEqual(-15, float(y_opt), delta=5)

    def test_ydoe_option(self):
        n_iter = 15
        fun = Branin(ndim=2)
        xlimits = fun.xlimits
        criterion = "LCB"  #'EI' or 'SBO' or 'LCB'
        random_state = 42
        design_space = DesignSpace(xlimits, seed=random_state)
        xdoe = FullFactorial(xlimits=xlimits)(10)
        ydoe = fun(xdoe)
        ego = EGO(
            xdoe=xdoe,
            ydoe=ydoe,
            n_iter=n_iter,
            criterion=criterion,
            surrogate=KRG(design_space=design_space, print_global=False),
            random_state=random_state,
        )
        _, y_opt, _, _, _ = ego.optimize(fun=fun)

        self.assertAlmostEqual(0.39, float(y_opt), delta=1)

    def test_find_best_point(self):
        fun = TestEGO.function_test_1d
        xlimits = np.array([[0.0, 25.0]])
        random_state = 42
        design_space = DesignSpace(xlimits, seed=random_state)
        xdoe = FullFactorial(xlimits=xlimits)(3)
        ydoe = fun(xdoe)
        ego = EGO(
            xdoe=xdoe,
            ydoe=ydoe,
            n_iter=1,
            criterion="LCB",
            surrogate=KRG(design_space=design_space, print_global=False),
            n_start=30,
            enable_tunneling=False,
            random_state=random_state,
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
        random_state = 42
        design_space = DesignSpace(fun.xlimits, seed=42)

        # Construction of the DOE
        sampling = LHS(xlimits=fun.xlimits, criterion="m", random_state=random_state)
        xdoe = sampling(20)
        ydoe = fun(xdoe)

        # Build the GEKPLS surrogate model
        n_comp = 2
        sm = GEKPLS(
            theta0=[1e-2] * n_comp,
            design_space=design_space,
            extra_points=1,
            eval_comp_treshold=0.8,
            print_prediction=False,
            n_comp=n_comp,
        )

        # Build the EGO optimizer and optimize
        ego = EGO(
            xdoe=xdoe,
            ydoe=ydoe,
            n_iter=5,
            criterion=criterion,
            surrogate=sm,
            n_start=30,
            enable_tunneling=False,
            random_state=random_state,
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
        random_state = 42
        design_space = DesignSpace(xlimits, seed=random_state)
        xdoe = FullFactorial(xlimits=xlimits)(3)
        ydoe = fun(xdoe)
        ego = EGO(
            xdoe=xdoe,
            ydoe=ydoe,
            n_iter=1,
            n_parallel=2,
            criterion="SBO",
            surrogate=KRG(design_space=design_space, print_global=False),
            n_start=30,
            random_state=random_state,
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

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 2, "too slow")
    def test_examples(self):
        self.run_ego_example()
        self.run_ego_parallel_example()
        self.run_ego_mixed_integer_example()

    @staticmethod
    def run_ego_example():
        import numpy as np
        from smt.applications import EGO
        from smt.surrogate_models import KRG
        from smt.utils.design_space import DesignSpace
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

        random_state = 42  # for reproducibility
        design_space = DesignSpace(xlimits, seed=random_state)
        xdoe = np.atleast_2d([0, 7, 25]).T
        n_doe = xdoe.size

        criterion = "EI"  #'EI' or 'SBO' or 'LCB'

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            surrogate=KRG(design_space=design_space, print_global=False),
            random_state=random_state,
        )

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
        from smt.applications.mixed_integer import MixedIntegerContext
        from smt.surrogate_models import MixIntKernelType
        from smt.utils.design_space import (
            DesignSpace,
            CategoricalVariable,
            FloatVariable,
            IntegerVariable,
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
            return y.reshape((-1, 1))

        n_iter = 15
        random_state = 42
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["blue", "red", "green"]),
                CategoricalVariable(["square", "circle"]),
                IntegerVariable(0, 2),
            ],
            seed=random_state,
        )

        criterion = "EI"  #'EI' or 'SBO' or 'LCB'
        qEI = "KBRand"
        sm = KRG(
            design_space=design_space,
            categorical_kernel=MixIntKernelType.GOWER,
            print_global=False,
        )
        mixint = MixedIntegerContext(design_space)
        n_doe = 3
        sampling = mixint.build_sampling_method(
            LHS, criterion="ese", random_state=random_state
        )
        xdoe = sampling(n_doe)
        ydoe = function_test_mixed_integer(xdoe)

        ego = EGO(
            n_iter=n_iter,
            criterion=criterion,
            xdoe=xdoe,
            ydoe=ydoe,
            surrogate=sm,
            qEI=qEI,
            n_parallel=2,
            random_state=random_state,
        )

        x_opt, y_opt, _, _, y_data = ego.optimize(fun=function_test_mixed_integer)
        print("Minimum in x={} with f(x)={:.1f}".format(x_opt, float(y_opt)))
        # print("Minimum in typed x={}".format(ego.mixint.cast_to_mixed_integer(x_opt)))

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
        from smt.applications.ego import Evaluator
        from smt.surrogate_models import KRG, DesignSpace

        import matplotlib.pyplot as plt

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

        random_state = 42
        design_space = DesignSpace(xlimits, seed=random_state)
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
            surrogate=KRG(design_space=design_space, print_global=False),
            n_parallel=n_parallel,
            qEI=qEI,
            n_start=n_start,
            evaluator=ParallelEvaluator(),
            random_state=random_state,
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
