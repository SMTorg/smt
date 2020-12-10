import unittest
import numpy as np
import matplotlib

matplotlib.use("Agg")

from smt.applications.mixed_integer import (
    MixedIntegerContext,
    MixedIntegerSamplingMethod,
    FLOAT,
    ENUM,
    INT,
    check_xspec_consistency,
    unfold_xlimits_with_continuous_limits,
    fold_with_enum_index,
    unfold_with_enum_mask,
    compute_unfolded_dimension,
    cast_to_enum_value,
    cast_to_mixed_integer,
)
from smt.problems import Sphere
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG


class TestMixedInteger(unittest.TestCase):
    def test_check_xspec_consistency(self):
        xtypes = [FLOAT, (ENUM, 3), INT]
        xlimits = [[-10, 10], ["blue", "red", "green"]]  # Bad dimension
        with self.assertRaises(ValueError):
            check_xspec_consistency(xtypes, xlimits)

        xtypes = [FLOAT, (ENUM, 3), INT]
        xlimits = [[-10, 10], ["blue", "red"], [-10, 10]]  # Bad enum
        with self.assertRaises(ValueError):
            check_xspec_consistency(xtypes, xlimits)

    def test_krg_mixed_3D(self):
        xtypes = [FLOAT, (ENUM, 3), INT]
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
        xtypes = [FLOAT, (ENUM, 2), (ENUM, 2), INT]
        xlimits = np.array([[-5, 5], ["2", "3"], ["4", "5"], [0, 2]])
        sampling = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="ese")
        doe = sampling(10)
        self.assertEqual((10, 4), doe.shape)

    def test_cast_to_mixed_integer(self):
        xtypes = [FLOAT, (ENUM, 2), (ENUM, 3), INT]
        xlimits = np.array(
            [[-5, 5], ["blue", "red"], ["short", "medium", "long"], [0, 2]],
            dtype="object",
        )
        x = np.array([1.5, 0, 2, 1])
        self.assertEqual(
            [1.5, "blue", "long", 1], cast_to_mixed_integer(xtypes, xlimits, x)
        )

    def run_mixed_integer_lhs_example(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import colors

        from smt.sampling_methods import LHS
        from smt.applications.mixed_integer import (
            FLOAT,
            INT,
            ENUM,
            MixedIntegerSamplingMethod,
        )

        xtypes = [FLOAT, (ENUM, 2)]
        xlimits = [[0.0, 4.0], ["blue", "red"]]
        sampling = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="ese")

        num = 40
        x = sampling(num)

        print(x.shape)

        cmap = colors.ListedColormap(xlimits[1])
        plt.scatter(x[:, 0], np.zeros(num), c=x[:, 1], cmap=cmap)
        plt.show()

    def run_mixed_integer_qp_example(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import QP
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel, INT

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])

        # xtypes = [FLOAT, INT, (ENUM, 3), (ENUM, 2)]
        # FLOAT means x1 continuous
        # INT means x2 integer
        # (ENUM, 3) means x3, x4 & x5 are 3 levels of the same categorical variable
        # (ENUM, 2) means x6 & x7 are 2 levels of the same categorical variable

        sm = MixedIntegerSurrogateModel(xtypes=[INT], xlimits=[[0, 4]], surrogate=QP())
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

        from smt.surrogate_models import KRG
        from smt.sampling_methods import LHS, Random
        from smt.applications.mixed_integer import MixedIntegerContext, FLOAT, INT, ENUM

        xtypes = [INT, FLOAT, (ENUM, 4)]
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
        print(xt)
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


if __name__ == "__main__":
    TestMixedInteger().run_mixed_integer_context_example()
