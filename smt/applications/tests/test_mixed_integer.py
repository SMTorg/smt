import unittest
import numpy as np
import matplotlib

matplotlib.use("Agg")

from smt.applications.mixed_integer import (
    MixedIntegerContext,
    MixedIntegerSamplingMethod,
)
from smt.utils.mixed_integer import (
    check_xspec_consistency,
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
from smt.surrogate_models import KRG, QP, FLOAT, ENUM, ORD


class TestMixedInteger(unittest.TestCase):
    def test_qp_mixed_2D_INT(self):
        xtypes = [FLOAT, ORD]
        xlimits = [[-10, 10], [-10, 10]]
        mixint = MixedIntegerContext(xtypes, xlimits)

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

    def test_krg_mixed_3D_INT(self):
        xtypes = [FLOAT, (ENUM, 3), ORD]
        xlimits = [[-10, 10], ["blue", "red", "green"], [-10, 10]]
        mixint = MixedIntegerContext(xtypes, xlimits)

        sm = mixint.build_surrogate_model(KRG(print_prediction=False))
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

    def test_check_xspec_consistency(self):
        xtypes = [FLOAT, (ENUM, 3), ORD]
        xlimits = [[-10, 10], ["blue", "red", "green"]]  # Bad dimension
        with self.assertRaises(ValueError):
            check_xspec_consistency(xtypes, xlimits)

        xtypes = [FLOAT, (ENUM, 3), ORD]
        xlimits = [[-10, 10], ["blue", "red"], [-10, 10]]  # Bad enum
        with self.assertRaises(ValueError):
            check_xspec_consistency(xtypes, xlimits)

        xtypes = [FLOAT, (ENUM, 2), (ENUM, 3), ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], ["0", "4", "3"]],
            dtype="object",
        )
        l = unfold_xlimits_with_continuous_limits(xtypes, xlimits)
        with self.assertRaises(ValueError):
            check_xspec_consistency(xtypes, xlimits)

    def test_krg_mixed_3D(self):
        xtypes = [FLOAT, (ENUM, 3), ORD]
        xlimits = [[-10, 10], ["blue", "red", "green"], [-10, 10]]
        mixint = MixedIntegerContext(xtypes, xlimits)

        sm = mixint.build_surrogate_model(KRG(print_prediction=False))
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
        xtypes = [FLOAT, (ENUM, 3), ORD]
        xlimits = [[-10, 10], ["blue", "red", "green"], [-10, 10]]
        mixint = MixedIntegerContext(xtypes, xlimits)
        with self.assertRaises(ValueError):
            sm = mixint.build_surrogate_model(
                KRG(print_prediction=False, poly="linear")
            )

    def test_qp_mixed_2D(self):
        xtypes = [FLOAT, ORD]
        xlimits = [[-10, 10], [-10, 10]]
        mixint = MixedIntegerContext(xtypes, xlimits)

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
        xtypes = [FLOAT, (ENUM, 2)]
        self.assertEqual(3, compute_unfolded_dimension(xtypes))

    def test_unfold_with_enum_mask(self):
        xtypes = [FLOAT, (ENUM, 2)]
        x = np.array([[1.5, 1], [1.5, 0], [1.5, 1]])
        expected = [[1.5, 0, 1], [1.5, 1, 0], [1.5, 0, 1]]
        self.assertListEqual(expected, unfold_with_enum_mask(xtypes, x).tolist())

    def test_unfold_with_enum_mask_with_enum_first(self):
        xtypes = [(ENUM, 2), FLOAT]
        x = np.array([[1, 1.5], [0, 1.5], [1, 1.5]])
        expected = [[0, 1, 1.5], [1, 0, 1.5], [0, 1, 1.5]]
        self.assertListEqual(expected, unfold_with_enum_mask(xtypes, x).tolist())

    def test_fold_with_enum_index(self):
        xtypes = [FLOAT, (ENUM, 2)]
        x = np.array([[1.5, 0, 1], [1.5, 1, 0], [1.5, 0, 1]])
        expected = [[1.5, 1], [1.5, 0], [1.5, 1]]
        self.assertListEqual(expected, fold_with_enum_index(xtypes, x).tolist())

    def test_fold_with_enum_index_with_list(self):
        xtypes = [FLOAT, (ENUM, 2)]
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
        xtypes = [FLOAT, (ENUM, 2), (ENUM, 2), ORD]
        xlimits = np.array([[-5, 5], ["2", "3"], ["4", "5"], [0, 2]])
        sampling = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="ese")
        doe = sampling(10)
        self.assertEqual((10, 4), doe.shape)

    def test_cast_to_mixed_integer(self):
        xtypes = [FLOAT, (ENUM, 2), (ENUM, 3), ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], [0, 2]],
            dtype="object",
        )
        x = np.array([1.5, 0, 2, 1.1])
        self.assertEqual(
            [1.5, "blue", "long", 1], cast_to_mixed_integer(xtypes, xlimits, x)
        )

    def test_encode_with_enum_index(self):
        xtypes = [FLOAT, (ENUM, 2), (ENUM, 3), ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], [0, 2]],
            dtype="object",
        )
        x = [1.5, "blue", "long", 1]
        self.assertEqual(
            np.array_equal(
                np.array([1.5, 0, 2, 1]),
                encode_with_enum_index(xtypes, xlimits, x),
            ),
            True,
        )

    def test_unfold_xlimits_with_continuous_limits(self):
        xtypes = [FLOAT, (ENUM, 2), (ENUM, 3), ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], [0, 2]],
            dtype="object",
        )
        l = unfold_xlimits_with_continuous_limits(xtypes, xlimits)
        self.assertEqual(
            np.array_equal(
                [[-5, 5], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 2]],
                unfold_xlimits_with_continuous_limits(xtypes, xlimits),
            ),
            True,
        )

    def test_unfold_xlimits_with_continuous_limits_and_ordinal_values(self):
        xtypes = [FLOAT, (ENUM, 2), (ENUM, 3), ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], ["0", "3", "4"]],
            dtype="object",
        )
        l = unfold_xlimits_with_continuous_limits(xtypes, xlimits)

        self.assertEqual(
            np.array_equal(
                [[-5, 5], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 4]],
                unfold_xlimits_with_continuous_limits(xtypes, xlimits),
            ),
            True,
        )

    def test_cast_to_discrete_values(self):
        xtypes = [FLOAT, (ENUM, 2), (ENUM, 3), ORD]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], [0, 4]],
            dtype="object",
        )
        x = np.array([[2.6, 0.3, 0.5, 0.25, 0.45, 0.85, 3.1]])

        self.assertEqual(
            np.array_equal(
                np.array([[2.6, 0, 1, 0, 0, 1, 3]]),
                cast_to_discrete_values(xtypes, xlimits, None, x),
            ),
            True,
        )

    def test_cast_to_discrete_values_with_smooth_rounding_ordinal_values(self):
        xtypes = [FLOAT, (ENUM, 2), (ENUM, 3), ORD]

        x = np.array([[2.6, 0.3, 0.5, 0.25, 0.45, 0.85, 3.1]])
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], ["0", "2", "4"]],
            dtype="object",
        )
        self.assertEqual(
            np.array_equal(
                np.array([[2.6, 0, 1, 0, 0, 1, 4]]),
                cast_to_discrete_values(xtypes, xlimits, None, x),
            ),
            True,
        )

    def test_cast_to_discrete_values_with_hard_rounding_ordinal_values(self):
        xtypes = [FLOAT, (ENUM, 2), (ENUM, 3), ORD]

        x = np.array([[2.6, 0.3, 0.5, 0.25, 0.45, 0.85, 3.1]])
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], ["0", "4"]],
            dtype="object",
        )
        self.assertEqual(
            np.array_equal(
                np.array([[2.6, 0, 1, 0, 0, 1, 4]]),
                cast_to_discrete_values(xtypes, xlimits, None, x),
            ),
            True,
        )

    def test_cast_to_discrete_values_with_non_integer_ordinal_values(self):
        xtypes = [FLOAT, (ENUM, 2), (ENUM, 3), ORD]

        x = np.array([[2.6, 0.3, 0.5, 0.25, 0.45, 0.85, 3.1]])
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], ["0", "3.5"]],
            dtype="object",
        )
        self.assertEqual(
            np.array_equal(
                np.array([[2.6, 0, 1, 0, 0, 1, 3.5]]),
                cast_to_discrete_values(xtypes, xlimits, None, x),
            ),
            True,
        )

    def run_mixed_integer_lhs_example(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import colors

        from smt.sampling_methods import LHS, FLOAT, ENUM
        from smt.applications.mixed_integer import MixedIntegerSamplingMethod

        xtypes = [FLOAT, (ENUM, 2)]
        xlimits = [[0.0, 4.0], ["blue", "red"]]
        sampling = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="ese")

        num = 40
        x = sampling(num)

        cmap = colors.ListedColormap(xlimits[1])
        plt.scatter(x[:, 0], np.zeros(num), c=x[:, 1], cmap=cmap)
        plt.show()

    def run_mixed_integer_qp_example(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import QP, ORD
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])

        # xtypes = [FLOAT, ORD, (ENUM, 3), (ENUM, 2)]
        # FLOAT means x1 continuous
        # ORD means x2 ordered
        # (ENUM, 3) means x3, x4 & x5 are 3 levels of the same categorical variable
        # (ENUM, 2) means x6 & x7 are 2 levels of the same categorical variable

        sm = MixedIntegerSurrogateModel(xtypes=[ORD], xlimits=[[0, 4]], surrogate=QP())
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
        from smt.surrogate_models import KRG, FLOAT, ORD, ENUM
        from smt.applications.mixed_integer import MixedIntegerContext

        xtypes = [ORD, FLOAT, (ENUM, 4)]
        xlimits = [[0, 5], [0.0, 4.0], ["blue", "red", "green", "yellow"]]

        def ftest(x):
            return (x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]) * (x[:, 2] + 1)

        # context to create consistent DOEs and surrogate
        mixint = MixedIntegerContext(xtypes, xlimits)

        # DOE for training
        lhs = mixint.build_sampling_method(LHS, criterion="ese")

        num = mixint.get_unfolded_dimension() * 5
        print("DOE point nb = {}".format(num))
        xt = lhs(num)
        yt = ftest(xt)

        # Surrogate
        sm = mixint.build_surrogate_model(KRG())
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

    def test_mixed_gower_2D(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        from smt.surrogate_models import KRG, ENUM, FLOAT, GOWER_KERNEL
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel

        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        xlimits = [["0.0", "1.0", " 2.0", "3.0", "4.0"], [-5, 5]]

        # Surrogate
        sm = MixedIntegerSurrogateModel(
            categorical_kernel=GOWER_KERNEL,
            xtypes=[(ENUM, 5), FLOAT],
            xlimits=xlimits,
            surrogate=KRG(theta0=[1e-2], corr="abs_exp"),
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

        i = 0
        for x in x_pred:
            print(i, x)
            i += 1
        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertTrue(np.abs(np.sum(np.array([y[20], y[50], y[95]]) - yt)) < 1e-6)
        self.assertTrue(np.abs(np.sum(np.array([yvar[20], yvar[50], yvar[95]]))) < 1e-6)

        self.assertEqual(np.shape(y), (105, 1))

    def test_mixed_homo_gaussian_2D(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        from smt.surrogate_models import KRG, ENUM, FLOAT, EXP_HOMO_HSPHERE_KERNEL
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel

        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        xlimits = [["0.0", "1.0", " 2.0", "3.0", "4.0"], [-5, 5]]

        # Surrogate
        sm = MixedIntegerSurrogateModel(
            categorical_kernel=EXP_HOMO_HSPHERE_KERNEL,
            xtypes=[(ENUM, 5), FLOAT],
            xlimits=xlimits,
            surrogate=KRG(theta0=[1e-2], corr="abs_exp"),
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

        i = 0
        for x in x_pred:
            print(i, x)
            i += 1
        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertTrue(np.abs(np.sum(np.array([y[20], y[50], y[95]]) - yt)) < 1e-6)
        self.assertTrue(np.abs(np.sum(np.array([yvar[20], yvar[50], yvar[95]]))) < 1e-6)

        self.assertEqual(np.shape(y), (105, 1))

    def test_mixed_homo_hyp_2D(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        from smt.surrogate_models import KRG, ENUM, FLOAT, HOMO_HSPHERE_KERNEL
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel

        xt = np.array([[0, 5], [2, -1], [4, 0.5]])
        yt = np.array([[0.0], [1.0], [1.5]])
        xlimits = [["0.0", "1.0", " 2.0", "3.0", "4.0"], [-5, 5]]

        # Surrogate
        sm = MixedIntegerSurrogateModel(
            categorical_kernel=HOMO_HSPHERE_KERNEL,
            xtypes=[(ENUM, 5), FLOAT],
            xlimits=xlimits,
            surrogate=KRG(theta0=[1e-2], corr="abs_exp"),
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

        i = 0
        for x in x_pred:
            print(i, x)
            i += 1
        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertTrue(np.abs(np.sum(np.array([y[20], y[50], y[95]]) - yt)) < 1e-6)
        self.assertTrue(np.abs(np.sum(np.array([yvar[20], yvar[50], yvar[95]]))) < 1e-6)

        self.assertEqual(np.shape(y), (105, 1))

    def test_mixed_homo_gaussian_3D_PLS(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        from smt.surrogate_models import KPLS, ENUM, FLOAT, EXP_HOMO_HSPHERE_KERNEL
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel

        xt = np.array([[0.5, 0, 5], [2, 3, 4], [5, 2, -1], [-2, 4, 0.5]])
        yt = np.array([[0.0], [3], [1.0], [1.5]])
        xlimits = [[-5, 5], ["0.0", "1.0", " 2.0", "3.0", "4.0"], [-5, 5]]

        # Surrogate
        sm = MixedIntegerSurrogateModel(
            categorical_kernel=EXP_HOMO_HSPHERE_KERNEL,
            xtypes=[FLOAT, (ENUM, 5), FLOAT],
            xlimits=xlimits,
            surrogate=KPLS(
                theta0=[1e-2], n_comp=1, cat_kernel_comps=[3], corr="squar_exp"
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

    def test_mixed_homo_gaussian_3D_PLS_cate(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        from smt.surrogate_models import KPLS, ENUM, FLOAT, EXP_HOMO_HSPHERE_KERNEL
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel

        xt = np.array([[0.5, 0, 5], [2, 3, 4], [5, 2, -1], [-2, 4, 0.5]])
        yt = np.array([[0.0], [3], [1.0], [1.5]])
        xlimits = [[-5, 5], ["0.0", "1.0", " 2.0", "3.0", "4.0"], [-5, 5]]

        # Surrogate
        sm = MixedIntegerSurrogateModel(
            categorical_kernel=EXP_HOMO_HSPHERE_KERNEL,
            xtypes=[FLOAT, (ENUM, 5), FLOAT],
            xlimits=xlimits,
            surrogate=KPLS(
                theta0=[1e-2], n_comp=2, cat_kernel_comps=[3], corr="abs_exp"
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

    def test_mixed_homo_hyp_3D_PLS_cate(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        from smt.surrogate_models import KPLS, ENUM, FLOAT, HOMO_HSPHERE_KERNEL
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel

        xt = np.array([[0.5, 0, 5], [2, 3, 4], [5, 2, -1], [-2, 4, 0.5]])
        yt = np.array([[0.0], [3], [1.0], [1.5]])
        xlimits = [[-5, 5], ["0.0", "1.0", " 2.0", "3.0", "4.0"], [-5, 5]]

        # Surrogate
        sm = MixedIntegerSurrogateModel(
            categorical_kernel=HOMO_HSPHERE_KERNEL,
            xtypes=[FLOAT, (ENUM, 5), FLOAT],
            xlimits=xlimits,
            surrogate=KPLS(
                theta0=[1e-2], n_comp=1, cat_kernel_comps=[3], corr="squar_exp"
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
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        from smt.surrogate_models import KPLS, ORD, ENUM, EXP_HOMO_HSPHERE_KERNEL
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel

        xt = np.array([[0.5, 0, 5], [2, 3, 4], [5, 2, -1], [-2, 4, 0.5]])
        yt = np.array([[0.0], [3], [1.0], [1.5]])
        xlimits = [
            ["0.0", "1.0", " 2.0", "3.0", "4.0"],
            [-5, 5],
            ["0.0", "1.0", " 2.0", "3.0"],
        ]

        # Surrogate
        sm = MixedIntegerSurrogateModel(
            categorical_kernel=EXP_HOMO_HSPHERE_KERNEL,
            xtypes=[(ENUM, 5), ORD, (ENUM, 4)],
            xlimits=xlimits,
            surrogate=KPLS(
                theta0=[1e-2], n_comp=1, cat_kernel_comps=[3, 2], corr="squar_exp"
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

        i = 0
        for x in x_pred:
            print(i, x)
            i += 1
        y = sm.predict_values(x_pred)
        yvar = sm.predict_variances(x_pred)

        # prediction are correct on known points
        self.assertTrue((np.abs(np.sum(np.array(sm.predict_values(xt) - yt)) < 1e-6)))
        self.assertTrue((np.abs(np.sum(np.array(sm.predict_variances(xt) - 0)) < 1e-6)))

    def test_mixed_gower(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import KRG, ENUM, FLOAT, GOWER_KERNEL
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel

        xt1 = np.array([[0, 0.0], [0, 2.0], [0, 4.0]])
        xt2 = np.array([[1, 0.0], [1, 2.0], [1, 3.0]])
        xt3 = np.array([[2, 1.0], [2, 2.0], [2, 4.0]])

        xt = np.concatenate((xt1, xt2, xt3), axis=0)
        xt[:, 1] = xt[:, 1].astype(np.float)
        yt1 = np.array([0.0, 9.0, 16.0])
        yt2 = np.array([0.0, -4, -13.0])
        yt3 = np.array([-10, 3, 11.0])

        yt = np.concatenate((yt1, yt2, yt3), axis=0)
        xlimits = [["Blue", "Red", "Green"], [0.0, 4.0]]
        xtypes = [(ENUM, 3), FLOAT]
        # Surrogate
        sm = MixedIntegerSurrogateModel(
            categorical_kernel=GOWER_KERNEL,
            xtypes=xtypes,
            xlimits=xlimits,
            surrogate=KRG(theta0=[1e-1], corr="squar_exp", n_start=20),
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

        axs[0].plot(xt1[:, 1].astype(np.float), yt1, "o", linestyle="None")
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
            xt2[:, 1].astype(np.float), yt2, marker="o", color="r", linestyle="None"
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
            xt3[:, 1].astype(np.float), yt3, marker="o", color="r", linestyle="None"
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

    def test_mixed_homo_gaussian(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import (
            KRG,
            ENUM,
            FLOAT,
            EXP_HOMO_HSPHERE_KERNEL,
        )
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel

        xt1 = np.array([[0, 0.0], [0, 2.0], [0, 4.0]])
        xt2 = np.array([[1, 0.0], [1, 2.0], [1, 3.0]])
        xt3 = np.array([[2, 1.0], [2, 2.0], [2, 4.0]])

        xt = np.concatenate((xt1, xt2, xt3), axis=0)
        xt[:, 1] = xt[:, 1].astype(np.float)
        yt1 = np.array([0.0, 9.0, 16.0])
        yt2 = np.array([0.0, -4, -13.0])
        yt3 = np.array([-10, 3, 11.0])

        yt = np.concatenate((yt1, yt2, yt3), axis=0)
        xlimits = [["Blue", "Red", "Green"], [0.0, 4.0]]
        xtypes = [(ENUM, 3), FLOAT]
        # Surrogate
        sm = MixedIntegerSurrogateModel(
            categorical_kernel=EXP_HOMO_HSPHERE_KERNEL,
            xtypes=xtypes,
            xlimits=xlimits,
            surrogate=KRG(theta0=[1e-1], corr="squar_exp", n_start=20),
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

        axs[0].plot(xt1[:, 1].astype(np.float), yt1, "o", linestyle="None")
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
            xt2[:, 1].astype(np.float), yt2, marker="o", color="r", linestyle="None"
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
            xt3[:, 1].astype(np.float), yt3, marker="o", color="r", linestyle="None"
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

    def test_mixed_homo_hyp(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import KRG, HOMO_HSPHERE_KERNEL, ENUM, FLOAT
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel

        xt1 = np.array([[0, 0.0], [0, 2.0], [0, 4.0]])
        xt2 = np.array([[1, 0.0], [1, 2.0], [1, 3.0]])
        xt3 = np.array([[2, 1.0], [2, 2.0], [2, 4.0]])

        xt = np.concatenate((xt1, xt2, xt3), axis=0)
        xt[:, 1] = xt[:, 1].astype(np.float)
        yt1 = np.array([0.0, 9.0, 16.0])
        yt2 = np.array([0.0, -4, -13.0])
        yt3 = np.array([-10, 3, 11.0])

        yt = np.concatenate((yt1, yt2, yt3), axis=0)
        xlimits = [["Blue", "Red", "Green"], [0.0, 4.0]]
        xtypes = [(ENUM, 3), FLOAT]
        # Surrogate
        sm = MixedIntegerSurrogateModel(
            categorical_kernel=HOMO_HSPHERE_KERNEL,
            xtypes=xtypes,
            xlimits=xlimits,
            surrogate=KRG(theta0=[1e-1], corr="squar_exp", n_start=20),
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

        axs[0].plot(xt1[:, 1].astype(np.float), yt1, "o", linestyle="None")
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
            xt2[:, 1].astype(np.float), yt2, marker="o", color="r", linestyle="None"
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
            xt3[:, 1].astype(np.float), yt3, marker="o", color="r", linestyle="None"
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
