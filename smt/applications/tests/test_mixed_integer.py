"""
Created on Tue Oct 12 10:48:01 2021
@author: psaves
"""

import itertools
import os
import unittest

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    NO_MATPLOTLIB = False
except ImportError:
    NO_MATPLOTLIB = True

from smt.applications.mixed_integer import (
    MixedIntegerContext,
    MixedIntegerKrigingModel,
    MixedIntegerSamplingMethod,
)
from smt.design_space import (
    CategoricalVariable,
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
)
from smt.problems import HierarchicalGoldstein, HierarchicalNeuralNetwork, Sphere
from smt.sampling_methods import LHS
from smt.surrogate_models import (
    KPLS,
    KPLSK,
    KRG,
    QP,
    MixHrcKernelType,
    MixIntKernelType,
)


class TestMixedInteger(unittest.TestCase):
    def test_periodic_mixed(self):
        # Objective function
        def f_obj(X):
            """
            s01 objective

            Parameters
            ----------
            point: array_like
                point to evaluate
            """
            design_space = DesignSpace(
                [
                    FloatVariable(0.0, 1.0),
                    CategoricalVariable(
                        values=[
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6",
                            "7",
                            "8",
                            "9",
                            "10",
                            "11",
                            "12",
                            "13",
                        ]
                    ),
                ]
            )

            PI = 3.14159265358979323846
            fail = False
            x = X[0]
            #  caté 1
            c = str(design_space.decode_values(X, i_dv=1)[1])
            x = np.abs(x)
            c1 = c == "1"
            c2 = c == "2"
            c3 = c == "3"
            c4 = c == "4"
            c5 = c == "5"
            c6 = c == "6"
            c7 = c == "7"
            c8 = c == "8"
            c9 = c == "9"
            c10 = c == "10"
            c11 = c == "11"
            c12 = c == "12"
            c13 = c == "13"

            if np.size(c1) == (
                np.sum(c1)
                + np.sum(c2)
                + np.sum(c3)
                + np.sum(c4)
                + np.sum(c5)
                + np.sum(c6)
                + np.sum(c7)
                + np.sum(c8)
                + np.sum(c9)
                + np.sum(c10)
                + np.sum(c11)
                + np.sum(c12)
                + np.sum(c13)
            ):
                u = (
                    1 * c1
                    + 2 * c2
                    + 3 * c3
                    + 4 * c4
                    + 5 * c5
                    + 6 * c6
                    + 7 * c7
                    + 8 * c8
                    + 9 * c9
                    + 10 * c10
                    + 11 * c11
                    + 12 * c12
                    + 13 * c13
                )
                hu = (PI * (0.4 + u / 15)) * (c10 + c11 + c12 + c13) - u / 20
                y = np.cos(3.5 * PI * x + hu)
            else:
                print("type error")
                print(X)
                fail = True
            return (y, fail)

        n_doe = 98

        design_space = DesignSpace(
            [
                FloatVariable(0.0, 1.0),
                CategoricalVariable(
                    values=[
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                    ]
                ),
            ]
        )

        sampling = MixedIntegerSamplingMethod(
            LHS,
            design_space,
            criterion="ese",
            seed=42,
            output_in_folded_space=True,
        )
        xdoe = sampling(n_doe)

        y_doe = [f_obj(xdoe[i])[0] for i in range(len(xdoe))]
        # Surrogate
        for m in MixIntKernelType:
            sm = MixedIntegerKrigingModel(
                surrogate=KRG(
                    design_space=design_space,
                    categorical_kernel=m,
                    theta0=[0.01],
                    hyper_opt="Cobyla",
                    corr="squar_sin_exp",
                    n_start=5,
                ),
            )

            sm.set_training_values(xdoe, np.array(y_doe))
            if m in [MixIntKernelType.EXP_HOMO_HSPHERE, MixIntKernelType.CONT_RELAX]:
                with self.assertRaises(ValueError):
                    sm.train()
            else:
                sm.train()

    def test_krg_mixed_3D(self):
        design_space = DesignSpace(
            [
                FloatVariable(-10, 10),
                CategoricalVariable(["blue", "red", "green"]),
                IntegerVariable(-10, 10),
            ]
        )

        mixint = MixedIntegerContext(design_space)

        sm = mixint.build_kriging_model(KRG(hyper_opt="Cobyla", print_prediction=False))
        sampling = mixint.build_sampling_method()

        fun = Sphere(ndim=3)
        xt = sampling(20)
        yt = fun(xt)
        sm.set_training_values(xt, yt)
        sm.train()

        eq_check = True
        for i in range(xt.shape[0]):
            if abs(float(xt[i, :][2]) - int(float(xt[i, :][2]))) > 10e-8:
                eq_check = False
            if not (xt[i, :][1] == 0 or xt[i, :][1] == 1 or xt[i, :][1] == 2):
                eq_check = False
        self.assertTrue(eq_check)

    def test_krg_mixed_3D_bad_regr(self):
        design_space = DesignSpace(
            [
                FloatVariable(-10, 10),
                CategoricalVariable(["blue", "red", "green"]),
                IntegerVariable(-10, 10),
            ]
        )

        mixint = MixedIntegerContext(design_space)
        with self.assertRaises(ValueError):
            _sm = mixint.build_kriging_model(
                KRG(hyper_opt="Cobyla", print_prediction=False, poly="linear")
            )

    def test_qp_mixed_2D_INT(self):
        design_space = DesignSpace(
            [
                FloatVariable(-10, 10),
                IntegerVariable(-10, 10),
            ]
        )

        mixint = MixedIntegerContext(design_space)
        sm = mixint.build_surrogate_model(QP(print_prediction=False))
        sampling = mixint.build_sampling_method()

        fun = Sphere(ndim=2)
        xt = sampling(10)
        yt = fun(xt)
        sm.set_training_values(xt, yt)
        sm.train()

        eq_check = True
        for i in range(xt.shape[0]):
            if abs(float(xt[i, :][1]) - int(float(xt[i, :][1]))) > 10e-8:
                eq_check = False
        self.assertTrue(eq_check)

    def test_compute_unfolded_dimension(self):
        design_space = DesignSpace(
            [
                FloatVariable(0, 1),
                CategoricalVariable(["A", "B"]),
            ]
        )
        assert design_space.get_unfolded_num_bounds().shape[0] == 3

    def test_unfold_with_enum_mask(self):
        x = np.array([[1.5, 1], [1.5, 0], [1.5, 1]])
        expected = [[1.5, 0, 1], [1.5, 1, 0], [1.5, 0, 1]]

        design_space = DesignSpace(
            [
                FloatVariable(1, 2),
                CategoricalVariable(["A", "B"]),
            ]
        )
        x_unfolded, _, _ = design_space.unfold_x(x)
        self.assertListEqual(expected, x_unfolded.tolist())

    def test_unfold_with_enum_mask_with_enum_first(self):
        x = np.array([[1, 1.5], [0, 1.5], [1, 1.5]])
        expected = [[0, 1, 1.5], [1, 0, 1.5], [0, 1, 1.5]]

        design_space = DesignSpace(
            [
                CategoricalVariable(["A", "B"]),
                FloatVariable(1, 2),
            ]
        )
        x_unfolded, _, _ = design_space.unfold_x(x)
        self.assertListEqual(expected, x_unfolded.tolist())

    def test_fold_with_enum_index(self):
        x = np.array([[1.5, 0, 1], [1.5, 1, 0], [1.5, 0, 1]])
        expected = [[1.5, 1], [1.5, 0], [1.5, 1]]

        design_space = DesignSpace(
            [
                FloatVariable(1, 2),
                CategoricalVariable(["A", "B"]),
            ]
        )
        x_folded, _ = design_space.fold_x(x)
        self.assertListEqual(expected, x_folded.tolist())

    def test_fold_with_enum_index_with_list(self):
        expected = [[1.5, 1]]
        x = np.array([1.5, 0, 1])

        design_space = DesignSpace(
            [
                FloatVariable(1, 2),
                CategoricalVariable(["A", "B"]),
            ]
        )
        x_folded, _ = design_space.fold_x(x)
        self.assertListEqual(expected, x_folded.tolist())
        x = [1.5, 0, 1]
        x_folded, _ = design_space.fold_x(x)
        self.assertListEqual(expected, x_folded.tolist())

    def test_cast_to_enum_value(self):
        design_space = DesignSpace(
            [
                FloatVariable(0, 4),
                CategoricalVariable(["blue", "red"]),
            ]
        )
        x = np.zeros((5, 2))
        x[:, 1] = [1, 1, 0, 1, 0]
        decoded = design_space.decode_values(x, i_dv=1)
        expected = ["red", "red", "blue", "red", "blue"]
        self.assertListEqual(expected, decoded)

    def test_unfolded_xlimits_type(self):
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["2", "3"]),
                CategoricalVariable(["4", "5"]),
                IntegerVariable(0, 2),
            ]
        )
        sampling = MixedIntegerSamplingMethod(LHS, design_space, criterion="ese")
        doe = sampling(10)
        self.assertEqual((10, 4), doe.shape)

    def test_unfold_xlimits_with_continuous_limits(self):
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["blue", "red"]),
                CategoricalVariable(["short", "medium", "long"]),
                IntegerVariable(0, 2),
            ]
        )

        unfolded_limits = design_space.get_unfolded_num_bounds()
        self.assertEqual(
            np.array_equal(
                [[-5, 5], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 2]],
                unfolded_limits,
            ),
            True,
        )

    def test_unfold_xlimits_with_continuous_limits_and_ordinal_values(self):
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["blue", "red"]),
                CategoricalVariable(["short", "medium", "long"]),
                OrdinalVariable(["0", "3", "4"]),
            ]
        )

        unfolded_limits = design_space.get_unfolded_num_bounds()
        self.assertEqual(
            np.array_equal(
                [[-5, 5], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 2]],
                unfolded_limits,
            ),
            True,
        )

    def test_cast_to_discrete_values(self):
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["blue", "red"]),
                CategoricalVariable(["short", "medium", "long"]),
                IntegerVariable(0, 4),
            ]
        )

        x = np.array([[2.6, 0.3, 0.5, 0.25, 0.45, 0.85, 3.1]])
        np.testing.assert_allclose(
            np.array([[2.6, 0, 1, 0, 0, 1, 3]]),
            design_space.correct_get_acting(x)[0],
            atol=1e-9,
        )

    def test_cast_to_discrete_values_with_smooth_rounding_ordinal_values(self):
        x = np.array([[2.6, 0.3, 0.5, 0.25, 0.45, 0.85, 1.1]])
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["blue", "red"]),
                CategoricalVariable(["short", "medium", "long"]),
                OrdinalVariable(["0", "2", "4"]),
            ]
        )

        np.testing.assert_allclose(
            np.array([[2.6, 0, 1, 0, 0, 1, 1]]),
            design_space.correct_get_acting(x)[0],
            atol=1e-9,
        )

    def test_cast_to_discrete_values_with_hard_rounding_ordinal_values(self):
        x = np.array([[2.6, 0.3, 0.5, 0.25, 0.45, 0.85, 0.9]])
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["blue", "red"]),
                CategoricalVariable(["short", "medium", "long"]),
                OrdinalVariable(["0", "4"]),
            ]
        )

        np.testing.assert_allclose(
            np.array([[2.6, 0, 1, 0, 0, 1, 1]]),
            design_space.correct_get_acting(x)[0],
            atol=1e-9,
        )

    def test_cast_to_discrete_values_with_non_integer_ordinal_values(self):
        x = np.array([[2.6, 0.3, 0.5, 0.25, 0.45, 0.85, 0.8]])
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["blue", "red"]),
                CategoricalVariable(["short", "medium", "long"]),
                OrdinalVariable(["0", "3.5"]),
            ]
        )
        np.testing.assert_allclose(
            np.array([[2.6, 0, 1, 0, 0, 1, 1]]),
            design_space.correct_get_acting(x)[0],
            atol=1e-9,
        )

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_examples(self):
        self.run_mixed_integer_lhs_example()
        self.run_mixed_integer_qp_example()
        self.run_mixed_integer_context_example()
        self.run_hierarchical_variables_Goldstein()
        self.run_mixed_discrete_design_space_example()
        self.run_mixed_gower_example()
        self.run_mixed_homo_gaussian_example()
        self.run_mixed_homo_hyp_example()
        # FIXME: this test should belong to smt_design_space_ext
        # but at the moment run_* code is used here to generate doc here in smt
        # if HAS_DESIGN_SPACE_EXT:
        #     self.run_mixed_cs_example()
        #     self.run_hierarchical_design_space_example()  # works only with config space impl

    def run_mixed_integer_lhs_example(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import colors

        from smt.applications.mixed_integer import MixedIntegerSamplingMethod
        from smt.design_space import (
            CategoricalVariable,
            DesignSpace,
            FloatVariable,
        )
        from smt.sampling_methods import LHS

        float_var = FloatVariable(0, 4)
        cat_var = CategoricalVariable(["blue", "red"])

        design_space = DesignSpace(
            [
                float_var,
                cat_var,
            ]
        )

        num = 40
        design_space.seed = 42
        samp = MixedIntegerSamplingMethod(
            LHS, design_space, criterion="ese", seed=design_space.seed
        )
        x, x_is_acting = samp(num, return_is_acting=True)

        cmap = colors.ListedColormap(cat_var.values)
        plt.scatter(x[:, 0], np.zeros(num), c=x[:, 1], cmap=cmap)
        plt.show()

    def run_mixed_integer_qp_example(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.applications.mixed_integer import MixedIntegerSurrogateModel
        from smt.design_space import DesignSpace, IntegerVariable
        from smt.surrogate_models import QP

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])

        # Specify the design space using the DesignSpace
        # class and various available variable types
        design_space = DesignSpace(
            [
                IntegerVariable(0, 4),
            ]
        )
        sm = MixedIntegerSurrogateModel(design_space=design_space, surrogate=QP())
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

    def run_mixed_integer_context_example(self):
        import matplotlib.pyplot as plt

        from smt.applications.mixed_integer import MixedIntegerContext
        from smt.design_space import (
            CategoricalVariable,
            DesignSpace,
            FloatVariable,
            IntegerVariable,
        )
        from smt.surrogate_models import KRG

        design_space = DesignSpace(
            [
                IntegerVariable(0, 5),
                FloatVariable(0.0, 4.0),
                CategoricalVariable(["blue", "red", "green", "yellow"]),
            ]
        )

        def ftest(x):
            return (x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]) * (x[:, 2] + 1)

        # Helper class for creating surrogate models
        mi_context = MixedIntegerContext(design_space)

        # DOE for training
        sampler = mi_context.build_sampling_method()

        num = mi_context.get_unfolded_dimension() * 5
        print("DOE point nb = {}".format(num))
        xt = sampler(num)
        yt = ftest(xt)

        # Surrogate
        sm = mi_context.build_kriging_model(KRG(hyper_opt="Cobyla"))
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        xv = sampler(50)
        yv = ftest(xv)
        yp = sm.predict_values(xv)

        plt.plot(yv, yv)
        plt.plot(yv, yp, "o")
        plt.xlabel("actual")
        plt.ylabel("prediction")

        plt.show()

    def test_hierarchical_variables_Goldstein(self):
        problem = HierarchicalGoldstein()
        ds = problem.design_space
        self.assertIsInstance(ds, DesignSpace)
        self.assertEqual(ds.n_dv, 11)

        x = np.array(
            [
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        y = problem(x)
        self.assertTrue(
            np.linalg.norm(
                y - np.array([50.75285716, 56.62074043, 50.97693309, 56.29235443])
            )
            < 1e-8
        )
        self.assertTrue(
            np.linalg.norm(
                problem.eval_is_acting.astype(int)
                - np.array(
                    [
                        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
                    ]
                )
            )
            < 1e-8
        )
        self.assertTrue(
            np.linalg.norm(
                problem.eval_x
                - np.array(
                    [
                        [0, 1, 1, 1, 50, 50, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 50, 1, 0, 1, 1, 1],
                        [2, 1, 1, 1, 50, 1, 1, 1, 0, 1, 1],
                        [3, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
                    ]
                )
            )
            < 1e-8
        )
        self.assertTrue(np.linalg.norm(y - problem(problem.eval_x)) < 1e-8)

        n_doe = 15
        ds.seed = 42
        samp = MixedIntegerSamplingMethod(LHS, ds, criterion="ese", seed=ds.seed)
        Xt, is_acting = samp(n_doe, return_is_acting=True)

        Yt = problem(Xt)

        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=ds,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                hierarchical_kernel=MixHrcKernelType.ARC_KERNEL,
                hyper_opt="Cobyla",
                theta0=[1e-2],
                corr="abs_exp",
                n_start=10,
            ),
        )
        sm.set_training_values(Xt, Yt, is_acting=is_acting)
        sm.train()
        y_s = sm.predict_values(Xt)[:, 0]
        pred_RMSE = np.linalg.norm(y_s - Yt) / len(Yt)

        y_sv = sm.predict_variances(Xt)[:, 0]
        var_RMSE = np.linalg.norm(y_sv) / len(Yt)
        self.assertLess(pred_RMSE, 1e-7)
        self.assertLess(var_RMSE, 1e-7)
        self.assertTrue(
            np.linalg.norm(
                sm.predict_values(
                    np.array(
                        [
                            [0.0, 1.0, 64.0, 4.0, 56.0, 37.0, 35.0, 1.0, 2.0, 1.0, 1.0],
                            [1.0, 0.0, 31.0, 92.0, 24.0, 3.0, 17.0, 1.0, 2.0, 1.0, 1.0],
                            [2.0, 1.0, 28.0, 60.0, 77.0, 66.0, 9.0, 0.0, 1.0, 1.0, 1.0],
                            [
                                3.0,
                                1.0,
                                50.0,
                                40.0,
                                99.0,
                                35.0,
                                51.0,
                                2.0,
                                1.0,
                                1.0,
                                2.0,
                            ],
                        ]
                    )
                )[:, 0]
                - sm.predict_values(
                    np.array(
                        [
                            [0.0, 1.0, 64.0, 4.0, 6.0, 7.0, 35.0, 1.0, 2.0, 1.0, 1.0],
                            [
                                1.0,
                                0.0,
                                31.0,
                                92.0,
                                24.0,
                                30.0,
                                17.0,
                                0.0,
                                2.0,
                                1.0,
                                1.0,
                            ],
                            [2.0, 1.0, 28.0, 60.0, 7.0, 66.0, 9.0, 0.0, 2.0, 1.0, 1.0],
                            [
                                3.0,
                                1.0,
                                50.0,
                                40.0,
                                99.0,
                                35.0,
                                51.0,
                                0.0,
                                0.0,
                                1.0,
                                2.0,
                            ],
                        ]
                    )
                )[:, 0]
            )
            < 1e-8
        )
        self.assertTrue(
            np.linalg.norm(
                sm.predict_values(
                    np.array(
                        [[1.0, 0.0, 31.0, 92.0, 24.0, 3.0, 17.0, 1.0, 2.0, 1.0, 1.0]]
                    )
                )
                - sm.predict_values(
                    np.array(
                        [[1.0, 0.0, 31.0, 92.0, 24.0, 3.0, 17.0, 1.0, 1.0, 1.0, 1.0]]
                    )
                )
            )
            > 1e-8
        )

    def run_mixed_discrete_design_space_example(self):
        import numpy as np

        from smt.applications.mixed_integer import MixedIntegerSamplingMethod
        from smt.design_space import (
            CategoricalVariable,
            DesignSpace,
            FloatVariable,
            IntegerVariable,
            OrdinalVariable,
        )
        from smt.sampling_methods import LHS

        ds = DesignSpace(
            [
                CategoricalVariable(
                    ["A", "B"]
                ),  # x0 categorical: A or B; order is not relevant
                OrdinalVariable(
                    ["C", "D", "E"]
                ),  # x1 ordinal: C, D or E; order is relevant
                IntegerVariable(
                    0, 2
                ),  # x2 integer between 0 and 2 (inclusive): 0, 1, 2
                FloatVariable(0, 1),  # c3 continuous between 0 and 1
            ]
        )

        # Sample the design space
        # Note: is_acting_sampled specifies for each design variable whether it is acting or not
        ds.seed = 42
        samp = MixedIntegerSamplingMethod(LHS, ds, criterion="ese", seed=ds.seed)
        x_sampled, is_acting_sampled = samp(100, return_is_acting=True)

        # Correct design vectors: round discrete variables, correct hierarchical variables
        x_corr, is_acting = ds.correct_get_acting(
            np.array(
                [
                    [0, 0, 2, 0.25],
                    [0, 2, 1, 0.75],
                ]
            )
        )
        print(is_acting)

    def run_hierarchical_design_space_example(self):
        import numpy as np
        from smt_design_space_ext import ConfigSpaceDesignSpaceImpl

        from smt.applications.mixed_integer import (
            MixedIntegerKrigingModel,
            MixedIntegerSamplingMethod,
        )
        from smt.design_space import (
            CategoricalVariable,
            FloatVariable,
            IntegerVariable,
            OrdinalVariable,
        )
        from smt.sampling_methods import LHS
        from smt.surrogate_models import KRG, MixHrcKernelType, MixIntKernelType

        ds = ConfigSpaceDesignSpaceImpl(
            [
                CategoricalVariable(
                    ["A", "B"]
                ),  # x0 categorical: A or B; order is not relevant
                OrdinalVariable(
                    ["C", "D", "E"]
                ),  # x1 ordinal: C, D or E; order is relevant
                IntegerVariable(
                    0, 2
                ),  # x2 integer between 0 and 2 (inclusive): 0, 1, 2
                FloatVariable(0, 1),  # c3 continuous between 0 and 1
            ]
        )

        # Declare that x1 is acting if x0 == A
        ds.declare_decreed_var(decreed_var=1, meta_var=0, meta_value="A")

        # Nested hierarchy is possible: activate x2 if x1 == C or D
        # Note: only if ConfigSpace is installed! pip install smt[cs]
        ds.declare_decreed_var(decreed_var=2, meta_var=1, meta_value=["C", "D"])

        # It is also possible to explicitly forbid two values from occurring simultaneously
        # Note: only if ConfigSpace is installed! pip install smt[cs]
        ds.add_value_constraint(
            var1=0, value1="A", var2=2, value2=[0, 1]
        )  # Forbid x0 == A && x2 == 0 or 1

        # For quantitative variables, it is possible to specify order relation
        ds.add_value_constraint(
            var1=2, value1="<", var2=3, value2=">"
        )  # Prevent x2 < x3

        # Sample the design space
        # Note: is_acting_sampled specifies for each design variable whether it is acting or not
        ds.seed = 42
        samp = MixedIntegerSamplingMethod(LHS, ds, criterion="ese", seed=ds.seed)
        Xt, is_acting_sampled = samp(100, return_is_acting=True)

        rng = np.random.default_rng(42)
        Yt = 4 * rng.random(100) - 2 + Xt[:, 0] + Xt[:, 1] - Xt[:, 2] - Xt[:, 3]
        # Correct design vectors: round discrete variables, correct hierarchical variables
        x_corr, is_acting = ds.correct_get_acting(
            np.array(
                [
                    [0, 0, 2, 0.25],
                    [0, 2, 1, 0.75],
                    [1, 2, 1, 0.66],
                ]
            )
        )

        # Observe the hierarchical behavior:
        assert np.all(
            is_acting
            == np.array(
                [
                    [True, True, True, True],
                    [
                        True,
                        True,
                        False,
                        True,
                    ],  # x2 is not acting if x1 != C or D (0 or 1)
                    [
                        True,
                        False,
                        False,
                        True,
                    ],  # x1 is not acting if x0 != A, and x2 is not acting because x1 is not acting
                ]
            )
        )
        assert np.all(
            x_corr
            == np.array(
                [
                    [0, 0, 2, 0.25],
                    [0, 2, 0, 0.75],
                    # x2 is not acting, so it is corrected ("imputed") to its non-acting value (0 for discrete vars)
                    [1, 0, 0, 0.66],  # x1 and x2 are imputed
                ]
            )
        )

        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=ds,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                hierarchical_kernel=MixHrcKernelType.ALG_KERNEL,
                theta0=[1e-2],
                hyper_opt="Cobyla",
                corr="abs_exp",
                n_start=5,
            ),
        )
        sm.set_training_values(Xt, Yt)
        sm.train()
        y_s = sm.predict_values(Xt)[:, 0]
        pred_RMSE = np.linalg.norm(y_s - Yt) / len(Yt)

        y_sv = sm.predict_variances(Xt)[:, 0]
        _var_RMSE = np.linalg.norm(y_sv) / len(Yt)
        assert pred_RMSE < 1e-7
        print("Pred_RMSE", pred_RMSE)

        self._sm = sm  # to be ignored: just used for automated test

    @unittest.skip(
        "Use smt design space extension capability, this test should be moved in smt_design_space_ext"
    )
    def test_hierarchical_design_space_example(self):
        self.run_hierarchical_design_space_example()
        np.testing.assert_almost_equal(
            self._sm.predict_values(
                np.array(
                    [
                        [0, 2, 1, 0.75],
                        [1, 2, 1, 0.66],
                        [0, 2, 1, 0.75],
                    ]
                )
            )[:, 0],
            self._sm.predict_values(
                np.array(
                    [
                        [0, 2, 2, 0.75],
                        [1, 1, 2, 0.66],
                        [0, 2, 0, 0.75],
                    ]
                )
            )[:, 0],
        )
        self.assertTrue(
            np.linalg.norm(
                self._sm.predict_values(np.array([[0, 0, 2, 0.25]]))
                - self._sm.predict_values(np.array([[0, 0, 2, 0.8]]))
            )
            > 1e-8
        )

    @unittest.skip(
        "Use smt design space extension capability, this test should be moved in smt_design_space_ext"
    )
    def test_hierarchical_design_space_example_all_categorical_decreed(self):
        ds = DesignSpace(
            [
                CategoricalVariable(
                    ["A", "B"]
                ),  # x0 categorical: A or B; order is not relevant
                CategoricalVariable(
                    ["C", "D", "E"]
                ),  # x1 ordinal: C, D or E; order is relevant
                CategoricalVariable(
                    ["tata", "tutu", "toto"]
                ),  # x2 integer between 0 and 2 (inclusive): 0, 1, 2
                FloatVariable(0, 1),  # c3 continuous between 0 and 1
            ]
        )

        # Declare that x1 is acting if x0 == A
        ds.declare_decreed_var(decreed_var=1, meta_var=0, meta_value="A")

        # Nested hierarchy is possible: activate x2 if x1 == C or D
        # Note: only if ConfigSpace is installed! pip install smt[cs]
        ds.declare_decreed_var(decreed_var=2, meta_var=1, meta_value=["C", "D"])

        # It is also possible to explicitly forbid two values from occurring simultaneously
        # Note: only if ConfigSpace is installed! pip install smt[cs]
        # ds.add_value_constraint(
        #    var1=0, value1="A", var2=2, value2=["tata","tutu"]
        # )  # Forbid x0 == A && x2 == 0 or 1

        # Sample the design space
        # Note: is_acting_sampled specifies for each design variable whether it is acting or not
        ds.seed = 42
        samp = MixedIntegerSamplingMethod(LHS, ds, criterion="ese", seed=ds.seed)
        Xt, is_acting_sampled = samp(100, return_is_acting=True)

        rng = np.random.default_rng(42)
        Yt = 4 * rng.random(100) - 2 + Xt[:, 0] + Xt[:, 1] - Xt[:, 2] - Xt[:, 3]
        # Correct design vectors: round discrete variables, correct hierarchical variables
        x_corr, is_acting = ds.correct_get_acting(
            np.array(
                [
                    [0, 0, 2, 0.25],
                    [0, 2, 1, 0.75],
                    [1, 2, 1, 0.66],
                ]
            )
        )

        # Observe the hierarchical behavior:
        np.testing.assert_array_equal(
            is_acting,
            np.array(
                [
                    [True, True, True, True],
                    [
                        True,
                        True,
                        False,
                        True,
                    ],  # x2 is not acting if x1 != C or D (0 or 1)
                    [
                        True,
                        False,
                        False,
                        True,
                    ],  # x1 is not acting if x0 != A, and x2 is not acting because x1 is not acting
                ]
            ),
        )

        np.testing.assert_array_equal(
            x_corr,
            np.array(
                [
                    [0, 0, 2, 0.25],
                    [0, 2, 0, 0.75],
                    # x2 is not acting, so it is corrected ("imputed") to its non-acting value (0 for discrete vars)
                    [1, 0, 0, 0.66],  # x1 and x2 are imputed
                ]
            ),
        )

        for mixint_kernel in [
            MixIntKernelType.CONT_RELAX,
            MixIntKernelType.GOWER,
            MixIntKernelType.COMPOUND_SYMMETRY,
            MixIntKernelType.EXP_HOMO_HSPHERE,
            MixIntKernelType.HOMO_HSPHERE,
        ]:
            sm = MixedIntegerKrigingModel(
                surrogate=KRG(
                    design_space=ds,
                    categorical_kernel=mixint_kernel,
                    hierarchical_kernel=MixHrcKernelType.ALG_KERNEL,
                    theta0=[1e-2],
                    hyper_opt="Cobyla",
                    corr="abs_exp",
                    n_start=5,
                ),
            )
            sm.set_training_values(Xt, Yt)
            sm.train()
            y_s = sm.predict_values(Xt)[:, 0]
            _pred_RMSE = np.linalg.norm(y_s - Yt) / len(Yt)
            self.assertLess(_pred_RMSE, 1e-6)
            y_sv = sm.predict_variances(Xt)[:, 0]
            _var_RMSE = np.linalg.norm(y_sv) / len(Yt)
            self.assertLess(_var_RMSE, 1e-6)
            np.testing.assert_almost_equal(
                sm.predict_values(
                    np.array(
                        [
                            [0, 2, 1, 0.75],
                            [1, 2, 1, 0.66],
                            [0, 2, 1, 0.75],
                        ]
                    )
                )[:, 0],
                sm.predict_values(
                    np.array(
                        [
                            [0, 2, 2, 0.75],
                            [1, 1, 2, 0.66],
                            [0, 2, 0, 0.75],
                        ]
                    )
                )[:, 0],
            )
            self.assertTrue(
                np.linalg.norm(
                    sm.predict_values(np.array([[0, 0, 2, 0.25]]))
                    - sm.predict_values(np.array([[0, 0, 2, 0.8]]))
                )
                > 1e-8
            )

    def run_hierarchical_variables_Goldstein(self):
        import numpy as np

        from smt.applications.mixed_integer import (
            MixedIntegerKrigingModel,
            MixedIntegerSamplingMethod,
        )
        from smt.design_space import (
            CategoricalVariable,
            DesignSpace,
            FloatVariable,
            IntegerVariable,
        )
        from smt.sampling_methods import LHS
        from smt.surrogate_models import KRG, MixHrcKernelType, MixIntKernelType

        def f_hv(X):
            import numpy as np

            def H(x1, x2, x3, x4, z3, z4, x5, cos_term):
                import numpy as np

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
            return np.array(y)

        design_space = DesignSpace(
            [
                CategoricalVariable(values=[0, 1, 2, 3]),  # meta
                IntegerVariable(0, 1),  # x1
                FloatVariable(0, 100),  # x2
                FloatVariable(0, 100),
                FloatVariable(0, 100),
                FloatVariable(0, 100),
                FloatVariable(0, 100),
                IntegerVariable(0, 2),  # x7
                IntegerVariable(0, 2),
                IntegerVariable(0, 2),
                IntegerVariable(0, 2),
            ]
        )

        # x4 is acting if meta == 1, 3
        design_space.declare_decreed_var(decreed_var=4, meta_var=0, meta_value=[1, 3])
        # x5 is acting if meta == 2, 3
        design_space.declare_decreed_var(decreed_var=5, meta_var=0, meta_value=[2, 3])
        # x7 is acting if meta == 0, 2
        design_space.declare_decreed_var(decreed_var=7, meta_var=0, meta_value=[0, 2])
        # x8 is acting if meta == 0, 1
        design_space.declare_decreed_var(decreed_var=8, meta_var=0, meta_value=[0, 1])

        # Sample from the design spaces, correctly considering hierarchy
        n_doe = 15
        design_space.seed = 42
        samp = MixedIntegerSamplingMethod(
            LHS, design_space, criterion="ese", seed=design_space.seed
        )
        Xt, Xt_is_acting = samp(n_doe, return_is_acting=True)

        Yt = f_hv(Xt)

        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=design_space,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                hierarchical_kernel=MixHrcKernelType.ALG_KERNEL,  # ALG or ARC
                theta0=[1e-2],
                hyper_opt="Cobyla",
                corr="abs_exp",
                n_start=5,
            ),
        )
        sm.set_training_values(Xt, Yt, is_acting=Xt_is_acting)
        sm.train()
        y_s = sm.predict_values(Xt)[:, 0]
        _pred_RMSE = np.linalg.norm(y_s - Yt) / len(Yt)

        y_sv = sm.predict_variances(Xt)[:, 0]
        _var_RMSE = np.linalg.norm(y_sv) / len(Yt)

    @unittest.skipIf(int(os.getenv("RUN_SLOW_TESTS", 0)) < 1, "too slow")
    def test_hierarchical_variables_NN(self):
        problem = HierarchicalNeuralNetwork()
        ds = problem.design_space
        self.assertEqual(ds.n_dv, 8)
        n_doe = 100

        ds_sampling = DesignSpace(ds.design_variables[1:])
        sampling = MixedIntegerSamplingMethod(
            LHS, ds_sampling, criterion="ese", seed=42
        )
        x_cont = sampling(3 * n_doe)

        xdoe1 = np.zeros((n_doe, 8))
        x_cont2 = x_cont[:n_doe, :5]
        xdoe1[:, 0] = np.zeros(n_doe)
        xdoe1[:, 1:6] = x_cont2
        ydoe1 = problem(xdoe1)

        xdoe1 = np.zeros((n_doe, 8))
        xdoe1[:, 0] = np.zeros(n_doe)
        xdoe1[:, 1:6] = x_cont2

        xdoe2 = np.zeros((n_doe, 8))
        x_cont2 = x_cont[n_doe : 2 * n_doe, :6]
        xdoe2[:, 0] = np.ones(n_doe)
        xdoe2[:, 1:7] = x_cont2
        ydoe2 = problem(xdoe2)

        xdoe2 = np.zeros((n_doe, 8))
        xdoe2[:, 0] = np.ones(n_doe)
        xdoe2[:, 1:7] = x_cont2

        xdoe3 = np.zeros((n_doe, 8))
        xdoe3[:, 0] = 2 * np.ones(n_doe)
        xdoe3[:, 1:] = x_cont[2 * n_doe :, :]
        ydoe3 = problem(xdoe3)

        Xt = np.concatenate((xdoe1, xdoe2, xdoe3), axis=0)
        Yt = np.concatenate((ydoe1, ydoe2, ydoe3), axis=0)
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=problem.design_space,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                hierarchical_kernel=MixHrcKernelType.ALG_KERNEL,
                theta0=[1e-2],
                hyper_opt="Cobyla",
                corr="abs_exp",
                n_start=5,
            ),
        )
        sm.set_training_values(Xt, Yt)
        sm.train()
        y_s = sm.predict_values(Xt)[:, 0]
        pred_RMSE = np.linalg.norm(y_s - Yt) / len(Yt)

        y_sv = sm.predict_variances(Xt)[:, 0]
        var_RMSE = np.linalg.norm(y_sv) / len(Yt)
        self.assertLess(pred_RMSE, 1e-7)
        self.assertLess(var_RMSE, 1e-7)
        np.testing.assert_almost_equal(
            sm.predict_values(
                np.array(
                    [
                        [0, -1, -2, 0, 0, 2, 0, 0],
                        [1, -1, -2, 1, 1, 2, 1, 0],
                        [2, -1, -2, 2, 2, 2, 1, -2],
                    ]
                )
            )[:, 0],
            sm.predict_values(
                np.array(
                    [
                        [0, -1, -2, 0, 0, 2, 10, 10],
                        [1, -1, -2, 1, 1, 2, 1, 10],
                        [2, -1, -2, 2, 2, 2, 1, -2],
                    ]
                )
            )[:, 0],
        )
        self.assertTrue(
            np.linalg.norm(
                sm.predict_values(np.array([[0, -1, -2, 0, 0, 2, 0, 0]]))
                - sm.predict_values(np.array([[0, -1, -2, 0, 0, 12, 10, 10]]))
            )
            > 1e-8
        )

    def test_mixed_gower_2D(self):
        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        design_space = DesignSpace(
            [
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
                FloatVariable(-5, 5),
            ]
        )

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=design_space,
                theta0=[1e-2],
                hyper_opt="Cobyla",
                corr="abs_exp",
                categorical_kernel=MixIntKernelType.GOWER,
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        x2 = np.linspace(-5, 5, 21)
        x1 = []
        for element in itertools.product(x, x2):
            x1.append(np.array(element))
        x_pred = np.array(x1)

        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertAlmostEqual(y[20, 0], 0)
        self.assertAlmostEqual(y[50, 0], 1)
        self.assertAlmostEqual(y[95, 0], 1.5)
        self.assertTrue(np.abs(np.sum(np.array([y[20], y[50], y[95]]) - yt)) < 1e-6)
        self.assertTrue(np.abs(np.sum(np.array([yvar[20], yvar[50], yvar[95]]))) < 1e-6)

        self.assertEqual(np.shape(y), (105, 1))

    def test_mixed_cs_2D(self):
        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        design_space = DesignSpace(
            [
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
                FloatVariable(-5, 5),
            ]
        )

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=design_space,
                theta0=[1e-2],
                hyper_opt="Cobyla",
                corr="abs_exp",
                categorical_kernel=MixIntKernelType.COMPOUND_SYMMETRY,
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        x2 = np.linspace(-5, 5, 21)
        x1 = []
        for element in itertools.product(x, x2):
            x1.append(np.array(element))
        x_pred = np.array(x1)

        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertAlmostEqual(y[20, 0], 0)
        self.assertAlmostEqual(y[50, 0], 1)
        self.assertAlmostEqual(y[95, 0], 1.5)
        self.assertTrue(np.abs(np.sum(np.array([y[20], y[50], y[95]]) - yt)) < 1e-6)
        self.assertTrue(np.abs(np.sum(np.array([yvar[20], yvar[50], yvar[95]]))) < 1e-6)

        self.assertEqual(np.shape(y), (105, 1))

    def test_mixed_CR_2D(self):
        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        design_space = DesignSpace(
            [
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
                FloatVariable(-5, 5),
            ]
        )

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=design_space,
                theta0=[1e-2],
                hyper_opt="Cobyla",
                corr="abs_exp",
                categorical_kernel=MixIntKernelType.GOWER,
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        x2 = np.linspace(-5, 5, 21)
        x1 = []
        for element in itertools.product(x, x2):
            x1.append(np.array(element))
        x_pred = np.array(x1)

        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertAlmostEqual(y[20, 0], 0)
        self.assertAlmostEqual(y[50, 0], 1)
        self.assertAlmostEqual(y[95, 0], 1.5)
        self.assertTrue(np.abs(np.sum(np.array([y[20], y[50], y[95]]) - yt)) < 1e-6)
        self.assertTrue(np.abs(np.sum(np.array([yvar[20], yvar[50], yvar[95]]))) < 1e-6)

        self.assertEqual(np.shape(y), (105, 1))

    def test_mixed_CR_PLS_noisy_2D(self):
        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        design_space = DesignSpace(
            [
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
                FloatVariable(-5, 5),
            ]
        )
        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KPLS(
                n_comp=1,
                eval_noise=True,
                design_space=design_space,
                theta0=[1e-2],
                categorical_kernel=MixIntKernelType.CONT_RELAX,
                hyper_opt="Cobyla",
                corr="abs_exp",
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        x2 = np.linspace(-5, 5, 21)
        x1 = []
        for element in itertools.product(x, x2):
            x1.append(np.array(element))
        x_pred = np.array(x1)

        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertTrue(np.abs(np.sum(np.array([y[20], y[50], y[95]]) - yt)) < 1e-6)
        self.assertTrue(np.abs(np.sum(np.array([yvar[20], yvar[50], yvar[95]]))) < 1e-6)

        self.assertEqual(np.shape(y), (105, 1))

    def test_mixed_CR_PLS_2D(self):
        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        design_space = DesignSpace(
            [
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
                FloatVariable(-5, 5),
            ]
        )
        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KPLS(
                n_comp=1,
                eval_noise=False,
                design_space=design_space,
                theta0=[1e-2],
                categorical_kernel=MixIntKernelType.CONT_RELAX,
                hyper_opt="Cobyla",
                corr="abs_exp",
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        x2 = np.linspace(-5, 5, 21)
        x1 = []
        for element in itertools.product(x, x2):
            x1.append(np.array(element))
        x_pred = np.array(x1)

        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertTrue(np.abs(np.sum(np.array([y[20], y[50], y[95]]) - yt)) < 1e-6)
        self.assertTrue(np.abs(np.sum(np.array([yvar[20], yvar[50], yvar[95]]))) < 1e-6)

        self.assertEqual(np.shape(y), (105, 1))

    def test_mixed_CR_KPLSK_2D(self):
        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        design_space = DesignSpace(
            [
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
                FloatVariable(-5, 5),
            ]
        )
        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KPLSK(
                n_comp=1,
                eval_noise=False,
                design_space=design_space,
                theta0=[1e-2],
                categorical_kernel=MixIntKernelType.CONT_RELAX,
                hyper_opt="Cobyla",
                corr="abs_exp",
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        x2 = np.linspace(-5, 5, 21)
        x1 = []
        for element in itertools.product(x, x2):
            x1.append(np.array(element))
        x_pred = np.array(x1)

        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertTrue(np.abs(np.sum(np.array([y[20], y[50], y[95]]) - yt)) < 1e-6)
        self.assertTrue(np.abs(np.sum(np.array([yvar[20], yvar[50], yvar[95]]))) < 1e-6)

        self.assertEqual(np.shape(y), (105, 1))

    def test_mixed_GD_KPLSK_2D(self):
        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        design_space = DesignSpace(
            [
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
                FloatVariable(-5, 5),
            ]
        )
        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KPLSK(
                n_comp=1,
                eval_noise=False,
                design_space=design_space,
                theta0=[1e-2],
                categorical_kernel=MixIntKernelType.GOWER,
                hyper_opt="Cobyla",
                corr="abs_exp",
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        x2 = np.linspace(-5, 5, 21)
        x1 = []
        for element in itertools.product(x, x2):
            x1.append(np.array(element))
        x_pred = np.array(x1)

        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertTrue(np.abs(np.sum(np.array([y[20], y[50], y[95]]) - yt)) < 1e-6)
        self.assertTrue(np.abs(np.sum(np.array([yvar[20], yvar[50], yvar[95]]))) < 1e-6)

        self.assertEqual(np.shape(y), (105, 1))

    def test_mixed_homo_gaussian_2D(self):
        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        design_space = DesignSpace(
            [
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
                FloatVariable(-5, 5),
            ]
        )

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=design_space,
                theta0=[1e-2],
                corr="abs_exp",
                hyper_opt="Cobyla",
                categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE,
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        x2 = np.linspace(-5, 5, 21)
        x1 = []
        for element in itertools.product(x, x2):
            x1.append(np.array(element))
        x_pred = np.array(x1)

        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertTrue(np.abs(np.sum(np.array([y[20], y[50], y[95]]) - yt)) < 1e-6)
        self.assertTrue(np.abs(np.sum(np.array([yvar[20], yvar[50], yvar[95]]))) < 1e-6)

        self.assertEqual(np.shape(y), (105, 1))

    def test_mixed_homo_hyp_2D(self):
        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        design_space = DesignSpace(
            [
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
                FloatVariable(-5, 5),
            ]
        )

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=design_space,
                theta0=[1e-2],
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                hyper_opt="Cobyla",
                corr="abs_exp",
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        x2 = np.linspace(-5, 5, 21)
        x1 = []
        for element in itertools.product(x, x2):
            x1.append(np.array(element))
        x_pred = np.array(x1)

        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertTrue(np.abs(np.sum(np.array([y[20], y[50], y[95]]) - yt)) < 1e-6)
        self.assertTrue(np.abs(np.sum(np.array([yvar[20], yvar[50], yvar[95]]))) < 1e-6)

        self.assertEqual(np.shape(y), (105, 1))

    def test_mixed_homo_gaussian_3D_PLS(self):
        xt = np.array([[0.5, 0, 5], [2, 3, 4], [5, 2, -1], [-2, 4, 0.5]])
        yt = np.array([[0.0], [3], [1.0], [1.5]])
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
                FloatVariable(-5, 5),
            ]
        )

        # Surrogate
        sm = KPLS(
            design_space=design_space,
            theta0=[1e-2],
            n_comp=1,
            categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE,
            cat_kernel_comps=[3],
            corr="squar_exp",
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        x2 = np.linspace(-5, 5, 21)
        x1 = []
        for element in itertools.product(x2, x, x2):
            x1.append(np.array(element))
        x_pred = np.array(x1)

        i = 0
        i += 1
        _y = sm.predict_values(x_pred)
        _yvar = sm.predict_variances(x_pred)

        self.assertTrue((np.abs(np.sum(np.array(sm.predict_values(xt) - yt)))) < 1e-6)
        self.assertTrue((np.abs(np.sum(np.array(sm.predict_variances(xt) - 0)))) < 1e-6)

    def test_mixed_homo_gaussian_3D_PLS_cate(self):
        xt = np.array([[0.5, 0, 5], [2, 3, 4], [5, 2, -1], [-2, 4, 0.5]])
        yt = np.array([[0.0], [3], [1.0], [1.5]])
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
                FloatVariable(-5, 5),
            ]
        )

        # Surrogate
        sm = KPLS(
            design_space=design_space,
            theta0=[1e-2],
            n_comp=2,
            hyper_opt="Cobyla",
            corr="abs_exp",
            cat_kernel_comps=[3],
            categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE,
        )

        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        x2 = np.linspace(-5, 5, 21)
        x1 = []
        for element in itertools.product(x2, x, x2):
            x1.append(np.array(element))
        x_pred = np.array(x1)

        i = 0
        i += 1
        _y = sm.predict_values(x_pred)
        _yvar = sm.predict_variances(x_pred)

        self.assertTrue((np.abs(np.sum(np.array(sm.predict_values(xt) - yt)))) < 1e-6)
        self.assertTrue((np.abs(np.sum(np.array(sm.predict_variances(xt) - 0)))) < 1e-6)

    def test_mixed_homo_hyp_3D_PLS_cate(self):
        xt = np.array([[0.5, 0, 5], [2, 3, 4], [5, 2, -1], [-2, 4, 0.5]])
        yt = np.array([[0.0], [3], [1.0], [1.5]])
        design_space = DesignSpace(
            [
                FloatVariable(-5, 5),
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
                FloatVariable(-5, 5),
            ]
        )

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KPLS(
                design_space=design_space,
                theta0=[1e-2],
                n_comp=1,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                cat_kernel_comps=[3],
                hyper_opt="Cobyla",
                corr="squar_exp",
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        x2 = np.linspace(-5, 5, 21)
        x1 = []
        for element in itertools.product(x2, x, x2):
            x1.append(np.array(element))
        x_pred = np.array(x1)

        i = 0
        i += 1
        _y = sm.predict_values(x_pred)
        _yvar = sm.predict_variances(x_pred)

        self.assertTrue((np.abs(np.sum(np.array(sm.predict_values(xt) - yt)))) < 1e-6)
        self.assertTrue((np.abs(np.sum(np.array(sm.predict_variances(xt) - 0)))) < 1e-6)

    def test_compound_hetero_noise_auto(self):
        xt1 = np.array([[0, 0.0], [0, 2.0], [0, 4.0]])
        xt2 = np.array([[1, 0.0], [1, 2.0], [1, 3.0]])
        xt3 = np.array([[2, 1.0], [2, 2.0], [2, 4.0]])

        xt = np.concatenate((xt1, xt2, xt3), axis=0)
        xt[:, 1] = xt[:, 1].astype(np.float64)
        yt1 = np.array([0.0, 9.0, 16.0])
        yt2 = np.array([0.0, -4, -13.0])
        yt3 = np.array([-10, 3, 11.0])
        yt = np.concatenate((yt1, yt2, yt3), axis=0)

        design_space = DesignSpace(
            [
                CategoricalVariable(["Blue", "Red", "Green"]),
                # OrdinalVariable([0,1,2]),
                FloatVariable(0, 4),
            ]
        )

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=design_space,
                categorical_kernel=MixIntKernelType.COMPOUND_SYMMETRY,
                theta0=[1e-1],
                corr="squar_exp",
                n_start=20,
                eval_noise=True,
                #     noise0 = [1.0,0.1,10,0.01,1.0,0.1,10,0.01,10],
                use_het_noise=True,
                hyper_opt="Cobyla",
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        n = 100
        x_cat1 = []
        x_cat2 = []
        x_cat3 = []

        for i in range(n):
            x_cat1.append(0)
            x_cat2.append(1)
            x_cat3.append(2)

        x_cont = np.linspace(0.0, 4.0, n)
        x1 = np.concatenate(
            (np.asarray(x_cat1).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )
        x2 = np.concatenate(
            (np.asarray(x_cat2).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )
        x3 = np.concatenate(
            (np.asarray(x_cat3).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )

        _y1 = sm.predict_values(x1)
        _y2 = sm.predict_values(x2)
        _y3 = sm.predict_values(x3)

        # estimated variance
        _s2_1 = sm.predict_variances(x1)
        _s2_2 = sm.predict_variances(x2)
        _s2_3 = sm.predict_variances(x3)

        self.assertEqual(
            np.array_equal(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                sm._surrogate.optimal_noise,
            ),
            True,
        )

    def test_mixed_homo_gaussian_3D_ord_cate(self):
        xt = np.array([[0, 5, 0], [2, 4, 3], [4, -1, 2], [2, 0.5, 1]])
        yt = np.array([[0.0], [3], [1.0], [1.5]])
        design_space = DesignSpace(
            [
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0", "4.0"]),
                FloatVariable(-5, 5),
                CategoricalVariable(["0.0", "1.0", " 2.0", "3.0"]),
            ]
        )

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KPLS(
                design_space=design_space,
                theta0=[1e-2],
                n_comp=1,
                categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE,
                cat_kernel_comps=[3, 2],
                hyper_opt="Cobyla",
                corr="squar_exp",
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        x = np.linspace(0, 4, 5)
        x2 = np.linspace(-5, 5, 21)
        x3 = np.linspace(0, 3, 4)
        x1 = []
        for element in itertools.product(x, x2, x3):
            x1.append(np.array(element))
        x_pred = np.array(x1)

        _y = sm.predict_values(x_pred)
        _yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertTrue((np.abs(np.sum(np.array(sm.predict_values(xt) - yt)) < 1e-6)))
        self.assertTrue((np.abs(np.sum(np.array(sm.predict_variances(xt) - 0)) < 1e-6)))

    def test_mixed_gower_3D(self):
        design_space = DesignSpace(
            [
                FloatVariable(-10, 10),
                IntegerVariable(-10, 10),
                IntegerVariable(-10, 10),
            ]
        )
        mixint = MixedIntegerContext(design_space)

        sm = mixint.build_kriging_model(
            KRG(
                categorical_kernel=MixIntKernelType.GOWER,
                print_prediction=False,
                hyper_opt="Cobyla",
            )
        )
        sampling = mixint.build_sampling_method()

        fun = Sphere(ndim=3)
        xt = sampling(10)
        yt = fun(xt)
        sm.set_training_values(xt, yt)
        sm.train()
        eq_check = True
        for i in range(xt.shape[0]):
            if abs(float(xt[i, :][1]) - int(float(xt[i, :][1]))) > 10e-8:
                eq_check = False
        self.assertTrue(eq_check)

    def run_mixed_gower_example(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.applications.mixed_integer import (
            MixedIntegerKrigingModel,
        )
        from smt.design_space import (
            CategoricalVariable,
            DesignSpace,
            FloatVariable,
        )
        from smt.surrogate_models import KRG, MixIntKernelType

        xt1 = np.array([[0, 0.0], [0, 2.0], [0, 4.0]])
        xt2 = np.array([[1, 0.0], [1, 2.0], [1, 3.0]])
        xt3 = np.array([[2, 1.0], [2, 2.0], [2, 4.0]])

        xt = np.concatenate((xt1, xt2, xt3), axis=0)
        xt[:, 1] = xt[:, 1].astype(np.float64)
        yt1 = np.array([0.0, 9.0, 16.0])
        yt2 = np.array([0.0, -4, -13.0])
        yt3 = np.array([-10, 3, 11.0])
        yt = np.concatenate((yt1, yt2, yt3), axis=0)

        design_space = DesignSpace(
            [
                CategoricalVariable(["Blue", "Red", "Green"]),
                FloatVariable(0, 4),
            ]
        )

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=design_space,
                categorical_kernel=MixIntKernelType.GOWER,
                theta0=[1e-1],
                hyper_opt="Cobyla",
                corr="squar_exp",
                n_start=20,
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        n = 100
        x_cat1 = []
        x_cat2 = []
        x_cat3 = []

        for i in range(n):
            x_cat1.append(0)
            x_cat2.append(1)
            x_cat3.append(2)

        x_cont = np.linspace(0.0, 4.0, n)
        x1 = np.concatenate(
            (np.asarray(x_cat1).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )
        x2 = np.concatenate(
            (np.asarray(x_cat2).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )
        x3 = np.concatenate(
            (np.asarray(x_cat3).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )

        y1 = sm.predict_values(x1)
        y2 = sm.predict_values(x2)
        y3 = sm.predict_values(x3)

        # estimated variance
        s2_1 = sm.predict_variances(x1)
        s2_2 = sm.predict_variances(x2)
        s2_3 = sm.predict_variances(x3)

        fig, axs = plt.subplots(3, figsize=(8, 6))

        axs[0].plot(xt1[:, 1].astype(np.float64), yt1, "o", linestyle="None")
        axs[0].plot(x_cont, y1, color="Blue")
        axs[0].fill_between(
            np.ravel(x_cont),
            np.ravel(y1 - 3 * np.sqrt(s2_1)),
            np.ravel(y1 + 3 * np.sqrt(s2_1)),
            color="lightgrey",
        )
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="upper left",
            bbox_to_anchor=[0, 1],
        )
        axs[1].plot(
            xt2[:, 1].astype(np.float64), yt2, marker="o", color="r", linestyle="None"
        )
        axs[1].plot(x_cont, y2, color="Red")
        axs[1].fill_between(
            np.ravel(x_cont),
            np.ravel(y2 - 3 * np.sqrt(s2_2)),
            np.ravel(y2 + 3 * np.sqrt(s2_2)),
            color="lightgrey",
        )
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="upper left",
            bbox_to_anchor=[0, 1],
        )
        axs[2].plot(
            xt3[:, 1].astype(np.float64), yt3, marker="o", color="r", linestyle="None"
        )
        axs[2].plot(x_cont, y3, color="Green")
        axs[2].fill_between(
            np.ravel(x_cont),
            np.ravel(y3 - 3 * np.sqrt(s2_3)),
            np.ravel(y3 + 3 * np.sqrt(s2_3)),
            color="lightgrey",
        )
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("y")
        axs[2].legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="upper left",
            bbox_to_anchor=[0, 1],
        )
        plt.tight_layout()
        plt.show()

    # FIXME: Used in SMT documentation but belongs to smt_design_space_ext domain
    def run_mixed_cs_example(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.applications.mixed_integer import (
            MixedIntegerKrigingModel,
        )
        from smt.design_space import (
            CategoricalVariable,
            DesignSpace,
            FloatVariable,
        )
        from smt.surrogate_models import KRG, MixIntKernelType

        xt1 = np.array([[0, 0.0], [0, 2.0], [0, 4.0]])
        xt2 = np.array([[1, 0.0], [1, 2.0], [1, 3.0]])
        xt3 = np.array([[2, 1.0], [2, 2.0], [2, 4.0]])

        xt = np.concatenate((xt1, xt2, xt3), axis=0)
        xt[:, 1] = xt[:, 1].astype(np.float64)
        yt1 = np.array([0.0, 9.0, 16.0])
        yt2 = np.array([0.0, -4, -13.0])
        yt3 = np.array([-10, 3, 11.0])
        yt = np.concatenate((yt1, yt2, yt3), axis=0)

        design_space = DesignSpace(
            [
                CategoricalVariable(["Blue", "Red", "Green"]),
                FloatVariable(0, 4),
            ]
        )

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=design_space,
                categorical_kernel=MixIntKernelType.COMPOUND_SYMMETRY,
                theta0=[1e-1],
                hyper_opt="Cobyla",
                corr="squar_exp",
                n_start=20,
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        n = 100
        x_cat1 = []
        x_cat2 = []
        x_cat3 = []

        for i in range(n):
            x_cat1.append(0)
            x_cat2.append(1)
            x_cat3.append(2)

        x_cont = np.linspace(0.0, 4.0, n)
        x1 = np.concatenate(
            (np.asarray(x_cat1).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )
        x2 = np.concatenate(
            (np.asarray(x_cat2).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )
        x3 = np.concatenate(
            (np.asarray(x_cat3).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )

        y1 = sm.predict_values(x1)
        y2 = sm.predict_values(x2)
        y3 = sm.predict_values(x3)

        # estimated variance
        s2_1 = sm.predict_variances(x1)
        s2_2 = sm.predict_variances(x2)
        s2_3 = sm.predict_variances(x3)

        fig, axs = plt.subplots(3, figsize=(8, 6))

        axs[0].plot(xt1[:, 1].astype(np.float64), yt1, "o", linestyle="None")
        axs[0].plot(x_cont, y1, color="Blue")
        axs[0].fill_between(
            np.ravel(x_cont),
            np.ravel(y1 - 3 * np.sqrt(s2_1)),
            np.ravel(y1 + 3 * np.sqrt(s2_1)),
            color="lightgrey",
        )
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="upper left",
            bbox_to_anchor=[0, 1],
        )
        axs[1].plot(
            xt2[:, 1].astype(np.float64), yt2, marker="o", color="r", linestyle="None"
        )
        axs[1].plot(x_cont, y2, color="Red")
        axs[1].fill_between(
            np.ravel(x_cont),
            np.ravel(y2 - 3 * np.sqrt(s2_2)),
            np.ravel(y2 + 3 * np.sqrt(s2_2)),
            color="lightgrey",
        )
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="upper left",
            bbox_to_anchor=[0, 1],
        )
        axs[2].plot(
            xt3[:, 1].astype(np.float64), yt3, marker="o", color="r", linestyle="None"
        )
        axs[2].plot(x_cont, y3, color="Green")
        axs[2].fill_between(
            np.ravel(x_cont),
            np.ravel(y3 - 3 * np.sqrt(s2_3)),
            np.ravel(y3 + 3 * np.sqrt(s2_3)),
            color="lightgrey",
        )
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("y")
        axs[2].legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="upper left",
            bbox_to_anchor=[0, 1],
        )
        plt.tight_layout()
        plt.show()

    # FIXME: Used in SMT documentation but belongs to smt_design_space_ext domain
    def run_mixed_homo_gaussian_example(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.applications.mixed_integer import MixedIntegerKrigingModel
        from smt.design_space import (
            CategoricalVariable,
            DesignSpace,
            FloatVariable,
        )
        from smt.surrogate_models import KRG, MixIntKernelType

        xt1 = np.array([[0, 0.0], [0, 2.0], [0, 4.0]])
        xt2 = np.array([[1, 0.0], [1, 2.0], [1, 3.0]])
        xt3 = np.array([[2, 1.0], [2, 2.0], [2, 4.0]])

        xt = np.concatenate((xt1, xt2, xt3), axis=0)
        xt[:, 1] = xt[:, 1].astype(np.float64)
        yt1 = np.array([0.0, 9.0, 16.0])
        yt2 = np.array([0.0, -4, -13.0])
        yt3 = np.array([-10, 3, 11.0])
        yt = np.concatenate((yt1, yt2, yt3), axis=0)

        design_space = DesignSpace(
            [
                CategoricalVariable(["Blue", "Red", "Green"]),
                FloatVariable(0, 4),
            ]
        )

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=design_space,
                theta0=[1e-1],
                hyper_opt="Cobyla",
                corr="squar_exp",
                n_start=20,
                categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE,
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        n = 100
        x_cat1 = []
        x_cat2 = []
        x_cat3 = []

        for i in range(n):
            x_cat1.append(0)
            x_cat2.append(1)
            x_cat3.append(2)

        x_cont = np.linspace(0.0, 4.0, n)
        x1 = np.concatenate(
            (np.asarray(x_cat1).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )
        x2 = np.concatenate(
            (np.asarray(x_cat2).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )
        x3 = np.concatenate(
            (np.asarray(x_cat3).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )

        y1 = sm.predict_values(x1)
        y2 = sm.predict_values(x2)
        y3 = sm.predict_values(x3)

        # estimated variance
        s2_1 = sm.predict_variances(x1)
        s2_2 = sm.predict_variances(x2)
        s2_3 = sm.predict_variances(x3)

        fig, axs = plt.subplots(3, figsize=(8, 6))

        axs[0].plot(xt1[:, 1].astype(np.float64), yt1, "o", linestyle="None")
        axs[0].plot(x_cont, y1, color="Blue")
        axs[0].fill_between(
            np.ravel(x_cont),
            np.ravel(y1 - 3 * np.sqrt(s2_1)),
            np.ravel(y1 + 3 * np.sqrt(s2_1)),
            color="lightgrey",
        )
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="upper left",
            bbox_to_anchor=[0, 1],
        )
        axs[1].plot(
            xt2[:, 1].astype(np.float64), yt2, marker="o", color="r", linestyle="None"
        )
        axs[1].plot(x_cont, y2, color="Red")
        axs[1].fill_between(
            np.ravel(x_cont),
            np.ravel(y2 - 3 * np.sqrt(s2_2)),
            np.ravel(y2 + 3 * np.sqrt(s2_2)),
            color="lightgrey",
        )
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="upper left",
            bbox_to_anchor=[0, 1],
        )
        axs[2].plot(
            xt3[:, 1].astype(np.float64), yt3, marker="o", color="r", linestyle="None"
        )
        axs[2].plot(x_cont, y3, color="Green")
        axs[2].fill_between(
            np.ravel(x_cont),
            np.ravel(y3 - 3 * np.sqrt(s2_3)),
            np.ravel(y3 + 3 * np.sqrt(s2_3)),
            color="lightgrey",
        )
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("y")
        axs[2].legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="upper left",
            bbox_to_anchor=[0, 1],
        )
        plt.tight_layout()
        plt.show()

    def run_mixed_homo_hyp_example(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from smt.applications.mixed_integer import MixedIntegerKrigingModel
        from smt.design_space import (
            CategoricalVariable,
            DesignSpace,
            FloatVariable,
        )
        from smt.surrogate_models import KRG, MixIntKernelType

        xt1 = np.array([[0, 0.0], [0, 2.0], [0, 4.0]])
        xt2 = np.array([[1, 0.0], [1, 2.0], [1, 3.0]])
        xt3 = np.array([[2, 1.0], [2, 2.0], [2, 4.0]])

        xt = np.concatenate((xt1, xt2, xt3), axis=0)
        xt[:, 1] = xt[:, 1].astype(np.float64)
        yt1 = np.array([0.0, 9.0, 16.0])
        yt2 = np.array([0.0, -4, -13.0])
        yt3 = np.array([-10, 3, 11.0])
        yt = np.concatenate((yt1, yt2, yt3), axis=0)

        design_space = DesignSpace(
            [
                CategoricalVariable(["Blue", "Red", "Green"]),
                FloatVariable(0, 4),
            ]
        )

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                design_space=design_space,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                theta0=[1e-1],
                hyper_opt="Cobyla",
                corr="squar_exp",
                n_start=20,
            ),
        )
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        n = 100
        x_cat1 = []
        x_cat2 = []
        x_cat3 = []

        for i in range(n):
            x_cat1.append(0)
            x_cat2.append(1)
            x_cat3.append(2)

        x_cont = np.linspace(0.0, 4.0, n)
        x1 = np.concatenate(
            (np.asarray(x_cat1).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )
        x2 = np.concatenate(
            (np.asarray(x_cat2).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )
        x3 = np.concatenate(
            (np.asarray(x_cat3).reshape(-1, 1), x_cont.reshape(-1, 1)), axis=1
        )

        y1 = sm.predict_values(x1)
        y2 = sm.predict_values(x2)
        y3 = sm.predict_values(x3)

        # estimated variance
        s2_1 = sm.predict_variances(x1)
        s2_2 = sm.predict_variances(x2)
        s2_3 = sm.predict_variances(x3)

        fig, axs = plt.subplots(3, figsize=(8, 6))

        axs[0].plot(xt1[:, 1].astype(np.float64), yt1, "o", linestyle="None")
        axs[0].plot(x_cont, y1, color="Blue")
        axs[0].fill_between(
            np.ravel(x_cont),
            np.ravel(y1 - 3 * np.sqrt(s2_1)),
            np.ravel(y1 + 3 * np.sqrt(s2_1)),
            color="lightgrey",
        )
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="upper left",
            bbox_to_anchor=[0, 1],
        )
        axs[1].plot(
            xt2[:, 1].astype(np.float64), yt2, marker="o", color="r", linestyle="None"
        )
        axs[1].plot(x_cont, y2, color="Red")
        axs[1].fill_between(
            np.ravel(x_cont),
            np.ravel(y2 - 3 * np.sqrt(s2_2)),
            np.ravel(y2 + 3 * np.sqrt(s2_2)),
            color="lightgrey",
        )
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="upper left",
            bbox_to_anchor=[0, 1],
        )
        axs[2].plot(
            xt3[:, 1].astype(np.float64), yt3, marker="o", color="r", linestyle="None"
        )
        axs[2].plot(x_cont, y3, color="Green")
        axs[2].fill_between(
            np.ravel(x_cont),
            np.ravel(y3 - 3 * np.sqrt(s2_3)),
            np.ravel(y3 + 3 * np.sqrt(s2_3)),
            color="lightgrey",
        )
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("y")
        axs[2].legend(
            ["Training data", "Prediction", "Confidence Interval 99%"],
            loc="upper left",
            bbox_to_anchor=[0, 1],
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    TestMixedInteger().run_mixed_integer_context_example()
    unittest.main()
