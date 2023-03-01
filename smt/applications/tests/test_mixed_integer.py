"""
Created on Tue Oct 12 10:48:01 2021
@author: psaves
"""

import unittest
import numpy as np
import matplotlib
import itertools

matplotlib.use("Agg")

from smt.utils.kriging import XSpecs

from smt.applications.mixed_integer import (
    MixedIntegerContext,
    MixedIntegerSamplingMethod,
    MixedIntegerKrigingModel,
)
from smt.utils.mixed_integer import (
    unfold_xlimits_with_continuous_limits,
    fold_with_enum_index,
    unfold_with_enum_mask,
    compute_unfolded_dimension,
    cast_to_enum_value,
    cast_to_mixed_integer,
    cast_to_discrete_values,
    encode_with_enum_index,
)
from smt.problems import Sphere
from smt.sampling_methods import LHS
from smt.surrogate_models import (
    KRG,
    KPLS,
    QP,
    XType,
    XRole,
    MixIntKernelType,
)


class TestMixedInteger(unittest.TestCase):
    def test_krg_mixed_3D_INT(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 3), XType.ORD]
        xlimits = [[-10, 10], ["blue", "red", "green"], [-10, 10]]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        mixint = MixedIntegerContext(xspecs=xspecs)

        sm = mixint.build_kriging_model(KRG(print_prediction=False))
        sampling = mixint.build_sampling_method(LHS, criterion="m")

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

    def test_krg_mixed_3D(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 3), XType.ORD]
        xlimits = [[-10, 10], ["blue", "red", "green"], [-10, 10]]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        mixint = MixedIntegerContext(xspecs=xspecs)

        sm = mixint.build_kriging_model(KRG(print_prediction=False))
        sampling = mixint.build_sampling_method(LHS, criterion="m")

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
        xtypes = [XType.FLOAT, (XType.ENUM, 3), XType.ORD]
        xlimits = [[-10, 10], ["blue", "red", "green"], [-10, 10]]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        mixint = MixedIntegerContext(xspecs=xspecs)
        with self.assertRaises(ValueError):
            sm = mixint.build_kriging_model(KRG(print_prediction=False, poly="linear"))

    def test_qp_mixed_2D_INT(self):
        xtypes = [XType.FLOAT, XType.ORD]
        xlimits = [[-10, 10], [-10, 10]]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        mixint = MixedIntegerContext(xspecs=xspecs)
        sm = mixint.build_surrogate_model(QP(print_prediction=False))
        sampling = mixint.build_sampling_method(LHS, criterion="m")

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
        xtypes = [XType.FLOAT, (XType.ENUM, 2)]
        self.assertEqual(3, compute_unfolded_dimension(xtypes))

    def test_unfold_with_enum_mask(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 2)]
        x = np.array([[1.5, 1], [1.5, 0], [1.5, 1]])
        expected = [[1.5, 0, 1], [1.5, 1, 0], [1.5, 0, 1]]
        self.assertListEqual(expected, unfold_with_enum_mask(xtypes, x).tolist())

    def test_unfold_with_enum_mask_with_enum_first(self):
        xtypes = [(XType.ENUM, 2), XType.FLOAT]
        x = np.array([[1, 1.5], [0, 1.5], [1, 1.5]])
        expected = [[0, 1, 1.5], [1, 0, 1.5], [0, 1, 1.5]]
        self.assertListEqual(expected, unfold_with_enum_mask(xtypes, x).tolist())

    def test_fold_with_enum_index(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 2)]
        x = np.array([[1.5, 0, 1], [1.5, 1, 0], [1.5, 0, 1]])
        expected = [[1.5, 1], [1.5, 0], [1.5, 1]]
        self.assertListEqual(expected, fold_with_enum_index(xtypes, x).tolist())

    def test_fold_with_enum_index_with_list(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 2)]
        expected = [[1.5, 1]]
        x = np.array([1.5, 0, 1])
        self.assertListEqual(expected, fold_with_enum_index(xtypes, x).tolist())
        x = [1.5, 0, 1]
        self.assertListEqual(expected, fold_with_enum_index(xtypes, x).tolist())

    def test_cast_to_enum_value(self):
        xlimits = [[0.0, 4.0], ["blue", "red"]]
        x_col = 1
        enum_indexes = [1, 1, 0, 1, 0]
        expected = ["red", "red", "blue", "red", "blue"]
        self.assertListEqual(expected, cast_to_enum_value(xlimits, x_col, enum_indexes))

    def test_unfolded_xlimits_type(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 2), (XType.ENUM, 2), XType.ORD]
        xlimits = np.array([[-5, 5], ["2", "3"], ["4", "5"], [0, 2]])
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)
        sampling = MixedIntegerSamplingMethod(LHS, xspecs, criterion="ese")
        doe = sampling(10)
        self.assertEqual((10, 4), doe.shape)

    def test_cast_to_mixed_integer(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 2), (XType.ENUM, 3), XType.ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], [0, 2]],
            dtype="object",
        )
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        x = np.array([1.5, 0, 2, 1.1])
        self.assertEqual([1.5, "blue", "long", 1], cast_to_mixed_integer(xspecs, x))

    def test_encode_with_enum_index(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 2), (XType.ENUM, 3), XType.ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], [0, 2]],
            dtype="object",
        )
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        x = [1.5, "blue", "long", 1]
        self.assertEqual(
            np.array_equal(
                np.array([1.5, 0, 2, 1]),
                encode_with_enum_index(xspecs, x),
            ),
            True,
        )

    def test_unfold_xlimits_with_continuous_limits(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 2), (XType.ENUM, 3), XType.ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], [0, 2]],
            dtype="object",
        )
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        l = unfold_xlimits_with_continuous_limits(xspecs)
        self.assertEqual(
            np.array_equal(
                [[-5, 5], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 2]],
                unfold_xlimits_with_continuous_limits(xspecs),
            ),
            True,
        )

    def test_unfold_xlimits_with_continuous_limits_and_ordinal_values(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 2), (XType.ENUM, 3), XType.ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], ["0", "3", "4"]],
            dtype="object",
        )
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        l = unfold_xlimits_with_continuous_limits(xspecs)

        self.assertEqual(
            np.array_equal(
                [[-5, 5], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 4]],
                unfold_xlimits_with_continuous_limits(xspecs),
            ),
            True,
        )

    def test_cast_to_discrete_values(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 2), (XType.ENUM, 3), XType.ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], [0, 4]],
            dtype="object",
        )
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        x = np.array([[2.6, 0.3, 0.5, 0.25, 0.45, 0.85, 3.1]])

        self.assertEqual(
            np.array_equal(
                np.array([[2.6, 0, 1, 0, 0, 1, 3]]),
                cast_to_discrete_values(xspecs, True, x),
            ),
            True,
        )

    def test_cast_to_discrete_values_with_smooth_rounding_ordinal_values(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 2), (XType.ENUM, 3), XType.ORD]

        x = np.array([[2.6, 0.3, 0.5, 0.25, 0.45, 0.85, 3.1]])
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], ["0", "2", "4"]],
            dtype="object",
        )
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        self.assertEqual(
            np.array_equal(
                np.array([[2.6, 0, 1, 0, 0, 1, 4]]),
                cast_to_discrete_values(xspecs, True, x),
            ),
            True,
        )

    def test_cast_to_discrete_values_with_hard_rounding_ordinal_values(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 2), (XType.ENUM, 3), XType.ORD]

        x = np.array([[2.6, 0.3, 0.5, 0.25, 0.45, 0.85, 3.1]])
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], ["0", "4"]],
            dtype="object",
        )
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        self.assertEqual(
            np.array_equal(
                np.array([[2.6, 0, 1, 0, 0, 1, 4]]),
                cast_to_discrete_values(xspecs, True, x),
            ),
            True,
        )

    def test_cast_to_discrete_values_with_non_integer_ordinal_values(self):
        xtypes = [XType.FLOAT, (XType.ENUM, 2), (XType.ENUM, 3), XType.ORD]

        x = np.array([[2.6, 0.3, 0.5, 0.25, 0.45, 0.85, 3.1]])
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], ["0", "3.5"]],
            dtype="object",
        )
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        self.assertEqual(
            np.array_equal(
                np.array([[2.6, 0, 1, 0, 0, 1, 3.5]]),
                cast_to_discrete_values(xspecs, True, x),
            ),
            True,
        )

    def test_examples(self):
        self.run_mixed_integer_lhs_example()
        self.run_mixed_integer_qp_example()
        self.run_mixed_integer_context_example()

    def run_mixed_integer_lhs_example(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import colors

        from smt.sampling_methods import LHS
        from smt.surrogate_models import XType, XSpecs
        from smt.applications.mixed_integer import MixedIntegerSamplingMethod

        xtypes = [XType.FLOAT, (XType.ENUM, 2)]
        xlimits = [[0.0, 4.0], ["blue", "red"]]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        sampling = MixedIntegerSamplingMethod(LHS, xspecs, criterion="ese")

        num = 40
        x = sampling(num)

        cmap = colors.ListedColormap(xlimits[1])
        plt.scatter(x[:, 0], np.zeros(num), c=x[:, 1], cmap=cmap)
        plt.show()

    def run_mixed_integer_qp_example(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import QP, XType, XSpecs
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])

        # xtypes = [XType.FLOAT, XType.ORD, (ENUM, 3), (ENUM, 2)]
        # XType.FLOAT means x1 continuous
        # XType.ORD means x2 ordered
        # (ENUM, 3) means x3, x4 & x5 are 3 levels of the same categorical variable
        # (ENUM, 2) means x6 & x7 are 2 levels of the same categorical variable
        xspecs = XSpecs(xtypes=[XType.ORD], xlimits=[[0, 4]])
        sm = MixedIntegerSurrogateModel(xspecs=xspecs, surrogate=QP())
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
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from mpl_toolkits.mplot3d import Axes3D

        from smt.sampling_methods import LHS, Random
        from smt.surrogate_models import KRG, XType, XSpecs
        from smt.applications.mixed_integer import MixedIntegerContext

        xtypes = [XType.ORD, XType.FLOAT, (XType.ENUM, 4)]
        xlimits = [[0, 5], [0.0, 4.0], ["blue", "red", "green", "yellow"]]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        def ftest(x):
            return (x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]) * (x[:, 2] + 1)

        # context to create consistent DOEs and surrogate
        mixint = MixedIntegerContext(xspecs=xspecs)

        # DOE for training
        lhs = mixint.build_sampling_method(LHS, criterion="ese")

        num = mixint.get_unfolded_dimension() * 5
        print("DOE point nb = {}".format(num))
        xt = lhs(num)
        yt = ftest(xt)

        # Surrogate
        sm = mixint.build_kriging_model(KRG())
        sm.set_training_values(xt, yt)
        sm.train()

        # DOE for validation
        rand = mixint.build_sampling_method(Random)
        xv = rand(50)
        yv = ftest(xv)
        yp = sm.predict_values(xv)

        plt.plot(yv, yv)
        plt.plot(yv, yp, "o")
        plt.xlabel("actual")
        plt.ylabel("prediction")

        plt.show()

    def test_hierarchical_variables_Goldstein(self):
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
            return np.array(y)

        xlimits = [
            ["6,7", "3,7", "4,6", "3,4"],  # meta1 ord
            [0, 1],  # 0
            [0, 100],  # 1
            [0, 100],  # 2
            [0, 100],  # 3
            [0, 100],  # 4
            [0, 100],  # 5
            [0, 2],  # 6
            [0, 2],  # 7
            [0, 2],  # 8
            [0, 2],  # 9
        ]
        xroles = [
            XRole.META,
            XRole.NEUTRAL,
            XRole.NEUTRAL,
            XRole.NEUTRAL,
            XRole.DECREED,
            XRole.DECREED,
            XRole.NEUTRAL,
            XRole.DECREED,
            XRole.DECREED,
            XRole.NEUTRAL,
            XRole.NEUTRAL,
        ]
        # z or x, cos?;          x1,x2,          x3, x4,        x5:cos,       z1,z2;            exp1,exp2

        xtypes = [
            (XType.ENUM, 4),
            XType.ORD,
            XType.FLOAT,
            XType.FLOAT,
            XType.FLOAT,
            XType.FLOAT,
            XType.FLOAT,
            XType.ORD,
            XType.ORD,
            XType.ORD,
            XType.ORD,
        ]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits, xroles=xroles)
        n_doe = 15
        sampling = MixedIntegerSamplingMethod(
            LHS, xspecs, criterion="ese", random_state=42
        )
        Xt = sampling(n_doe)
        Yt = f_hv(Xt)

        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                xspecs=xspecs,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                theta0=[1e-2],
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
        self.assertTrue(pred_RMSE < 1e-7)
        print("Pred_RMSE", pred_RMSE)
        self.assertTrue(var_RMSE < 1e-7)
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

    def test_hierarchical_variables_NN(self):
        def f_neu(x1, x2, x3, x4):
            if x4 == 0:
                return 2 * x1 + x2 - 0.5 * x3
            if x4 == 1:
                return -x1 + 2 * x2 - 0.5 * x3
            if x4 == 2:
                return -x1 + x2 + 0.5 * x3

        def f1(x1, x2, x3, x4, x5):
            return f_neu(x1, x2, x3, x4) + x5**2

        def f2(x1, x2, x3, x4, x5, x6):
            return f_neu(x1, x2, x3, x4) + (x5**2) + 0.3 * x6

        def f3(x1, x2, x3, x4, x5, x6, x7):
            return f_neu(x1, x2, x3, x4) + (x5**2) + 0.3 * x6 - 0.1 * x7**3

        def f(X):
            y = []
            for x in X:
                if x[0] == 1:
                    y.append(f1(x[1], x[2], x[3], x[4], x[5]))
                elif x[0] == 2:
                    y.append(f2(x[1], x[2], x[3], x[4], x[5], x[6]))
                elif x[0] == 3:
                    y.append(f3(x[1], x[2], x[3], x[4], x[5], x[6], x[7]))
            return np.array(y)

        xlimits = [
            [1, 3],  # meta ord
            [-5, -2],
            [-5, -1],
            ["8", "16", "32", "64", "128", "256"],
            ["ReLU", "SELU", "ISRLU"],
            [0.0, 5.0],  # decreed m=1
            [0.0, 5.0],  # decreed m=2
            [0.0, 5.0],  # decreed m=3
        ]
        xtypes = [
            XType.ORD,
            XType.FLOAT,
            XType.FLOAT,
            XType.ORD,
            (XType.ENUM, 3),
            XType.ORD,
            XType.ORD,
            XType.ORD,
        ]
        xroles = [
            XRole.META,
            XRole.NEUTRAL,
            XRole.NEUTRAL,
            XRole.NEUTRAL,
            XRole.NEUTRAL,
            XRole.DECREED,
            XRole.DECREED,
            XRole.DECREED,
        ]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits, xroles=xroles)
        n_doe = 100

        xspecs_samp = XSpecs(xtypes=xtypes[1:], xlimits=xlimits[1:])

        sampling = MixedIntegerSamplingMethod(
            LHS, xspecs_samp, criterion="ese", random_state=42
        )
        x_cont = sampling(3 * n_doe)

        xdoe1 = np.zeros((n_doe, 6))
        x_cont2 = x_cont[:n_doe, :5]
        xdoe1[:, 0] = np.ones(n_doe)
        xdoe1[:, 1:] = x_cont2
        ydoe1 = f(xdoe1)

        xdoe1 = np.zeros((n_doe, 8))
        xdoe1[:, 0] = np.ones(n_doe)
        xdoe1[:, 1:6] = x_cont2

        xdoe2 = np.zeros((n_doe, 7))
        x_cont2 = x_cont[n_doe : 2 * n_doe, :6]
        xdoe2[:, 0] = 2 * np.ones(n_doe)
        xdoe2[:, 1:7] = x_cont2
        ydoe2 = f(xdoe2)

        xdoe2 = np.zeros((n_doe, 8))
        xdoe2[:, 0] = 2 * np.ones(n_doe)
        xdoe2[:, 1:7] = x_cont2

        xdoe3 = np.zeros((n_doe, 8))
        xdoe3[:, 0] = 3 * np.ones(n_doe)
        xdoe3[:, 1:] = x_cont[2 * n_doe :, :]
        ydoe3 = f(xdoe3)

        Xt = np.concatenate((xdoe1, xdoe2, xdoe3), axis=0)
        Yt = np.concatenate((ydoe1, ydoe2, ydoe3), axis=0)
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                xspecs=xspecs,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                theta0=[1e-2],
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
        self.assertTrue(pred_RMSE < 1e-7)
        print("Pred_RMSE", pred_RMSE)
        self.assertTrue(var_RMSE < 1e-7)
        self.assertTrue(
            np.linalg.norm(
                sm.predict_values(
                    np.array(
                        [
                            [1, -1, -2, 8, 0, 2, 0, 0],
                            [2, -1, -2, 16, 1, 2, 1, 0],
                            [3, -1, -2, 32, 2, 2, 1, -2],
                        ]
                    )
                )[:, 0]
                - sm.predict_values(
                    np.array(
                        [
                            [1, -1, -2, 8, 0, 2, 10, 10],
                            [2, -1, -2, 16, 1, 2, 1, 10],
                            [3, -1, -2, 32, 2, 2, 1, -2],
                        ]
                    )
                )[:, 0]
            )
            < 1e-8
        )
        self.assertTrue(
            np.linalg.norm(
                sm.predict_values(np.array([[1, -1, -2, 8, 0, 2, 0, 0]]))
                - sm.predict_values(np.array([[1, -1, -2, 8, 0, 12, 10, 10]]))
            )
            > 1e-8
        )

    def test_mixed_gower_2D(self):
        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        xlimits = [["0.0", "1.0", " 2.0", "3.0", "4.0"], [-5, 5]]
        xtypes = [(XType.ENUM, 5), XType.FLOAT]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)

        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                xspecs=xspecs,
                theta0=[1e-2],
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
        self.assertTrue(np.abs(np.sum(np.array([y[20], y[50], y[95]]) - yt)) < 1e-6)
        self.assertTrue(np.abs(np.sum(np.array([yvar[20], yvar[50], yvar[95]]))) < 1e-6)

        self.assertEqual(np.shape(y), (105, 1))

    def test_mixed_homo_gaussian_2D(self):
        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        xlimits = [["0.0", "1.0", " 2.0", "3.0", "4.0"], [-5, 5]]
        xtypes = [(XType.ENUM, 5), XType.FLOAT]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)
        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                xspecs=xspecs,
                theta0=[1e-2],
                corr="abs_exp",
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
        xlimits = [["0.0", "1.0", " 2.0", "3.0", "4.0"], [-5, 5]]
        xtypes = [(XType.ENUM, 5), XType.FLOAT]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)
        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                xspecs=xspecs,
                theta0=[1e-2],
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
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
        xlimits = [[-5, 5], ["0.0", "1.0", " 2.0", "3.0", "4.0"], [-5, 5]]
        xtypes = [XType.FLOAT, (XType.ENUM, 5), XType.FLOAT]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)
        # Surrogate
        sm = surrogate = KPLS(
            xspecs=xspecs,
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
        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        self.assertTrue((np.abs(np.sum(np.array(sm.predict_values(xt) - yt)))) < 1e-6)
        self.assertTrue((np.abs(np.sum(np.array(sm.predict_variances(xt) - 0)))) < 1e-6)

    def test_mixed_homo_gaussian_3D_PLS_cate(self):
        xt = np.array([[0.5, 0, 5], [2, 3, 4], [5, 2, -1], [-2, 4, 0.5]])
        yt = np.array([[0.0], [3], [1.0], [1.5]])
        xlimits = [[-5, 5], ["0.0", "1.0", " 2.0", "3.0", "4.0"], [-5, 5]]
        xtypes = [XType.FLOAT, (XType.ENUM, 5), XType.FLOAT]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)
        # Surrogate
        sm = KPLS(
            xspecs=xspecs,
            theta0=[1e-2],
            n_comp=2,
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
        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        self.assertTrue((np.abs(np.sum(np.array(sm.predict_values(xt) - yt)))) < 1e-6)
        self.assertTrue((np.abs(np.sum(np.array(sm.predict_variances(xt) - 0)))) < 1e-6)

    def test_mixed_homo_hyp_3D_PLS_cate(self):
        xt = np.array([[0.5, 0, 5], [2, 3, 4], [5, 2, -1], [-2, 4, 0.5]])
        yt = np.array([[0.0], [3], [1.0], [1.5]])
        xlimits = [[-5, 5], ["0.0", "1.0", " 2.0", "3.0", "4.0"], [-5, 5]]
        xtypes = [XType.FLOAT, (XType.ENUM, 5), XType.FLOAT]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)
        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KPLS(
                xspecs=xspecs,
                theta0=[1e-2],
                n_comp=1,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                cat_kernel_comps=[3],
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
        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        self.assertTrue((np.abs(np.sum(np.array(sm.predict_values(xt) - yt)))) < 1e-6)
        self.assertTrue((np.abs(np.sum(np.array(sm.predict_variances(xt) - 0)))) < 1e-6)

    def test_mixed_homo_gaussian_3D_ord_cate(self):
        xt = np.array([[0.5, 0, 5], [2, 3, 4], [5, 2, -1], [-2, 4, 0.5]])
        yt = np.array([[0.0], [3], [1.0], [1.5]])
        xlimits = [
            ["0.0", "1.0", " 2.0", "3.0", "4.0"],
            [-5, 5],
            ["0.0", "1.0", " 2.0", "3.0"],
        ]
        xtypes = [(XType.ENUM, 5), XType.ORD, (XType.ENUM, 4)]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)
        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KPLS(
                xspecs=xspecs,
                theta0=[1e-2],
                n_comp=1,
                categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE,
                cat_kernel_comps=[3, 2],
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

        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertTrue((np.abs(np.sum(np.array(sm.predict_values(xt) - yt)) < 1e-6)))
        self.assertTrue((np.abs(np.sum(np.array(sm.predict_variances(xt) - 0)) < 1e-6)))

    def test_mixed_gower_3D(self):
        xtypes = [XType.FLOAT, XType.ORD, XType.ORD]
        xlimits = [[-10, 10], [-10, 10], [-10, 10]]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)
        mixint = MixedIntegerContext(xspecs=xspecs)

        sm = mixint.build_kriging_model(
            KRG(categorical_kernel=MixIntKernelType.GOWER, print_prediction=False)
        )
        sampling = mixint.build_sampling_method(LHS, criterion="m")

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

    def test_examples(self):
        self.run_mixed_gower_example()
        self.run_mixed_homo_gaussian_example()
        self.run_mixed_homo_hyp_example()

    def run_mixed_gower_example(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import KRG, XType, XSpecs, MixIntKernelType
        from smt.applications.mixed_integer import MixedIntegerKrigingModel

        xt1 = np.array([[0, 0.0], [0, 2.0], [0, 4.0]])
        xt2 = np.array([[1, 0.0], [1, 2.0], [1, 3.0]])
        xt3 = np.array([[2, 1.0], [2, 2.0], [2, 4.0]])

        xt = np.concatenate((xt1, xt2, xt3), axis=0)
        xt[:, 1] = xt[:, 1].astype(np.float64)
        yt1 = np.array([0.0, 9.0, 16.0])
        yt2 = np.array([0.0, -4, -13.0])
        yt3 = np.array([-10, 3, 11.0])

        yt = np.concatenate((yt1, yt2, yt3), axis=0)
        xlimits = [["Blue", "Red", "Green"], [0.0, 4.0]]
        xtypes = [(XType.ENUM, 3), XType.FLOAT]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)
        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                xspecs=xspecs,
                categorical_kernel=MixIntKernelType.GOWER,
                theta0=[1e-1],
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

    def run_mixed_homo_gaussian_example(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import KRG, XType, XSpecs, MixIntKernelType
        from smt.applications.mixed_integer import MixedIntegerKrigingModel

        xt1 = np.array([[0, 0.0], [0, 2.0], [0, 4.0]])
        xt2 = np.array([[1, 0.0], [1, 2.0], [1, 3.0]])
        xt3 = np.array([[2, 1.0], [2, 2.0], [2, 4.0]])

        xt = np.concatenate((xt1, xt2, xt3), axis=0)
        xt[:, 1] = xt[:, 1].astype(np.float64)
        yt1 = np.array([0.0, 9.0, 16.0])
        yt2 = np.array([0.0, -4, -13.0])
        yt3 = np.array([-10, 3, 11.0])

        yt = np.concatenate((yt1, yt2, yt3), axis=0)
        xlimits = [["Blue", "Red", "Green"], [0.0, 4.0]]
        xtypes = [(XType.ENUM, 3), XType.FLOAT]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)
        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                xspecs=xspecs,
                theta0=[1e-1],
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
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import KRG, XType, XSpecs, MixIntKernelType
        from smt.applications.mixed_integer import MixedIntegerKrigingModel

        xt1 = np.array([[0, 0.0], [0, 2.0], [0, 4.0]])
        xt2 = np.array([[1, 0.0], [1, 2.0], [1, 3.0]])
        xt3 = np.array([[2, 1.0], [2, 2.0], [2, 4.0]])

        xt = np.concatenate((xt1, xt2, xt3), axis=0)
        xt[:, 1] = xt[:, 1].astype(np.float64)
        yt1 = np.array([0.0, 9.0, 16.0])
        yt2 = np.array([0.0, -4, -13.0])
        yt3 = np.array([-10, 3, 11.0])

        yt = np.concatenate((yt1, yt2, yt3), axis=0)
        xlimits = [["Blue", "Red", "Green"], [0.0, 4.0]]
        xtypes = [(XType.ENUM, 3), XType.FLOAT]
        xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)
        # Surrogate
        sm = MixedIntegerKrigingModel(
            surrogate=KRG(
                xspecs=xspecs,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                theta0=[1e-1],
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
