import unittest
import numpy as np
from smt.applications.mixed_integer import (
    MixedIntegerContext,
    FLOAT,
    ENUM,
    INT,
    check_xspec_consistency,
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

    def test_krg_mixed_5D(self):
        xtypes = [FLOAT, (ENUM, 3), INT]
        xlimits = [[-10, 10], ["blue", "red", "green"], [-10, 10]]
        mixint = MixedIntegerContext(xtypes, xlimits)

        sm = mixint.build_surrogate(KRG(print_prediction=False))
        sampling = mixint.build_sampling_method(LHS, criterion="m")

        fun = Sphere(ndim=5)
        xt = sampling(20)
        yt = fun(xt)
        sm.set_training_values(xt, yt)
        sm.train()

        x_out = mixint.fold_with_enum_indexes(xt)
        eq_check = True
        for i in range(x_out.shape[0]):
            if abs(float(x_out[i, :][2]) - int(float(x_out[i, :][2]))) > 10e-8:
                eq_check = False
            if not (x_out[i, :][1] == 0 or x_out[i, :][1] == 1 or x_out[i, :][1] == 2):
                eq_check = False
        self.assertTrue(eq_check)

    def test_mixed_integer_lhs(self):
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        import matplotlib.pyplot as plt

        from smt.sampling_methods import LHS
        from smt.applications.mixed_integer import (
            FLOAT,
            INT,
            ENUM,
            MixedIntegerSamplingMethod,
        )

        xtypes = [(ENUM, 2), FLOAT]
        xlimits = [["blue", "red"], [0.0, 4.0]]
        sampling = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="ese")

        num = 40
        x = sampling(num)

        print(x.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], "o")
        ax.set_xlabel("x0 blue (1) or not (0)")
        ax.set_ylabel("x1 red (1) or not (0)")
        ax.set_zlabel("x2 float")
        plt.show()

    def test_mixed_integer_qp(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.surrogate_models import QP
        from smt.applications.mixed_integer import MixedIntegerSurrogate, INT

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])

        # xtypes = [FLOAT, INT, (ENUM, 3), (ENUM, 2)]
        # FLOAT means x1 continuous
        # INT means x2 integer
        # (ENUM, 3) means x3, x4 & x5 are 3 levels of the same categorical variable
        # (ENUM, 2) means x6 & x7 are 2 levels of the same categorical variable

        sm = MixedIntegerSurrogate(xtypes=[INT], xlimits=[[0, 4]], surrogate=QP())
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
