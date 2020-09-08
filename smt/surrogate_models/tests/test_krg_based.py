"""
Author: Remi Lafage <<remi.lafage@onera.fr>>

This package is distributed under New BSD license.
"""

import unittest
import numpy as np
from smt.surrogate_models.krg_based import KrgBased


class TestKrgBased(unittest.TestCase):
    def test_theta0_default_init(self):
        krg = KrgBased()
        krg.set_training_values(np.array([[1, 2, 3]]), np.array([[1]]))
        krg._check_param()
        self.assertTrue(np.array_equal(krg.options["theta0"], [1e-2, 1e-2, 1e-2]))

    def test_theta0_one_dim_init(self):
        krg = KrgBased(theta0=[2e-2])
        krg.set_training_values(np.array([[1, 2, 3]]), np.array([[1]]))
        krg._check_param()
        self.assertTrue(np.array_equal(krg.options["theta0"], [2e-2, 2e-2, 2e-2]))

    def test_theta0_erroneous_init(self):
        krg = KrgBased(theta0=[2e-2, 1e-2])
        krg.set_training_values(np.array([[1, 2]]), np.array([[1]]))  # correct
        krg._check_param()
        krg.set_training_values(np.array([[1, 2, 3]]), np.array([[1]]))  # erroneous
        self.assertRaises(ValueError, krg._check_param)

    def test_krg_mixed_2D(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from smt.problems import Sphere
        from smt.sampling_methods import LHS
        from smt.surrogate_models import KRG

        vartype = ["cont", "int"]
        sm = KRG(vartype=vartype, print_prediction=False)
        fun = Sphere(ndim=2)
        xlimits_relaxed = sm._relax_limits(fun.xlimits, dim=2)
        sampling = LHS(xlimits=xlimits_relaxed, criterion="m")
        xt = sampling(20)
        xt = sm._project_values(xt)
        yt = fun(xt)
        sm.set_training_values(xt, yt)
        sm.train()
        X = np.arange(xlimits_relaxed[0, 0], xlimits_relaxed[0, 1], 0.25)
        Y = np.arange(xlimits_relaxed[1, 0], xlimits_relaxed[1, 1], 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros((X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = sm.predict_values(
                    np.hstack((X[i, j], Y[i, j])).reshape((1, 2))
                )
        x_out = sm._assign_labels(xt, fun.xlimits)
        Equality_check = True
        for i in range(x_out.shape[0]):
            if abs(float(x_out[i, :][1]) - int(float(x_out[i, :][1]))) > 10e-8:
                Equality_check = False
        self.assertTrue(Equality_check)

    def test_krg_mixed_4D(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from smt.problems import Sphere
        from smt.sampling_methods import LHS
        from smt.surrogate_models import KRG

        vartype = ["cont", ("cate", 2), "int"]
        xlimits = [[-10, 10], ["blue", "red"], [-10, -10]]
        sm = KRG(vartype=vartype, print_prediction=False)
        fun = Sphere(ndim=4)
        xlimits_relaxed = sm._relax_limits(xlimits, dim=4)
        sampling = LHS(xlimits=xlimits_relaxed, criterion="m")
        xt = sampling(20)
        xt = sm._project_values(xt)
        yt = fun(xt)
        sm.set_training_values(xt, yt)
        sm.train()
        x_out = sm._assign_labels(xt, xlimits)
        Equality_check = True
        for i in range(x_out.shape[0]):
            if abs(float(x_out[i, :][2]) - int(float(x_out[i, :][2]))) > 10e-8:
                Equality_check = False
            if not (x_out[i, :][1] == "blue" or x_out[i, :][1] == "red"):
                Equality_check = False
        self.assertTrue(Equality_check)

    def test_krg_mixed_5D(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from smt.problems import Sphere
        from smt.sampling_methods import LHS
        from smt.surrogate_models import KRG

        vartype = ["cont", ("cate", 3), "int"]
        xlimits = [[-10, 10], ["blue", "red", "green"], [-10, -10]]
        sm = KRG(vartype=vartype, print_prediction=False)
        fun = Sphere(ndim=5)
        xlimits_relaxed = sm._relax_limits(xlimits, dim=5)
        sampling = LHS(xlimits=xlimits_relaxed, criterion="m")
        xt = sampling(20)
        xt = sm._project_values(xt)
        yt = fun(xt)
        sm.set_training_values(xt, yt)
        sm.train()
        x_out = sm._assign_labels(xt, xlimits)
        Equality_check = True
        for i in range(x_out.shape[0]):
            if abs(float(x_out[i, :][2]) - int(float(x_out[i, :][2]))) > 10e-8:
                Equality_check = False
            if not (
                x_out[i, :][1] == "blue"
                or x_out[i, :][1] == "red"
                or x_out[i, :][1] == "green"
            ):
                Equality_check = False
        self.assertTrue(Equality_check)

    def test_kpls_mixed_2D(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from smt.problems import Sphere
        from smt.sampling_methods import LHS
        from smt.surrogate_models import KPLS

        vartype = ["cont", "int"]
        sm = KPLS(vartype=vartype, print_prediction=False)
        fun = Sphere(ndim=2)
        xlimits_relaxed = sm._relax_limits(fun.xlimits, dim=2)
        sampling = LHS(xlimits=xlimits_relaxed, criterion="m")
        xt = sampling(20)
        xt = sm._project_values(xt)
        yt = fun(xt)
        sm.set_training_values(xt, yt)
        sm.train()
        X = np.arange(xlimits_relaxed[0, 0], xlimits_relaxed[0, 1], 0.25)
        Y = np.arange(xlimits_relaxed[1, 0], xlimits_relaxed[1, 1], 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros((X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = sm.predict_values(
                    np.hstack((X[i, j], Y[i, j])).reshape((1, 2))
                )
        x_out = sm._assign_labels(xt, fun.xlimits)
        Equality_check = True
        for i in range(x_out.shape[0]):
            if abs(float(x_out[i, :][1]) - int(float(x_out[i, :][1]))) > 10e-8:
                Equality_check = False
        self.assertTrue(Equality_check)

    def test_kpls_mixed_4D(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from smt.problems import Sphere
        from smt.sampling_methods import LHS
        from smt.surrogate_models import KPLS

        vartype = ["cont", ("cate", 2), "int"]
        xlimits = [[-10, 10], ["blue", "red"], [-10, -10]]
        sm = KPLS(vartype=vartype, print_prediction=False)
        fun = Sphere(ndim=4)
        xlimits_relaxed = sm._relax_limits(xlimits, dim=4)
        sampling = LHS(xlimits=xlimits_relaxed, criterion="m")
        xt = sampling(20)
        xt = sm._project_values(xt)
        yt = fun(xt)
        sm.set_training_values(xt, yt)
        sm.train()
        x_out = sm._assign_labels(xt, xlimits)
        Equality_check = True
        for i in range(x_out.shape[0]):
            if abs(float(x_out[i, :][2]) - int(float(x_out[i, :][2]))) > 10e-8:
                Equality_check = False
            if not (x_out[i, :][1] == "blue" or x_out[i, :][1] == "red"):
                Equality_check = False
        self.assertTrue(Equality_check)

    def test_kplsk_mixed_2D(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from smt.problems import Sphere
        from smt.sampling_methods import LHS
        from smt.surrogate_models import KPLSK

        vartype = ["cont", "int"]
        sm = KPLSK(vartype=vartype, print_prediction=False)
        fun = Sphere(ndim=2)
        xlimits_relaxed = sm._relax_limits(fun.xlimits, dim=2)
        sampling = LHS(xlimits=xlimits_relaxed, criterion="m")
        xt = sampling(20)
        xt = sm._project_values(xt)
        yt = fun(xt)
        sm.set_training_values(xt, yt)
        sm.train()
        X = np.arange(xlimits_relaxed[0, 0], xlimits_relaxed[0, 1], 0.25)
        Y = np.arange(xlimits_relaxed[1, 0], xlimits_relaxed[1, 1], 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros((X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = sm.predict_values(
                    np.hstack((X[i, j], Y[i, j])).reshape((1, 2))
                )
        x_out = sm._assign_labels(xt, fun.xlimits)
        Equality_check = True
        for i in range(x_out.shape[0]):
            if abs(float(x_out[i, :][1]) - int(float(x_out[i, :][1]))) > 10e-8:
                Equality_check = False
        self.assertTrue(Equality_check)

    def test_kplsk_mixed_4D(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from smt.problems import Sphere
        from smt.sampling_methods import LHS
        from smt.surrogate_models import KPLSK

        vartype = ["cont", ("cate", 2), "int"]
        xlimits = [[-10, 10], ["blue", "red"], [-10, -10]]
        sm = KPLSK(vartype=vartype, print_prediction=False)
        fun = Sphere(ndim=4)
        xlimits_relaxed = sm._relax_limits(xlimits, dim=4)
        sampling = LHS(xlimits=xlimits_relaxed, criterion="m")
        xt = sampling(20)
        xt = sm._project_values(xt)
        yt = fun(xt)
        sm.set_training_values(xt, yt)
        sm.train()
        x_out = sm._assign_labels(xt, xlimits)
        Equality_check = True
        for i in range(x_out.shape[0]):
            if abs(float(x_out[i, :][2]) - int(float(x_out[i, :][2]))) > 10e-8:
                Equality_check = False
            if not (x_out[i, :][1] == "blue" or x_out[i, :][1] == "red"):
                Equality_check = False
        self.assertTrue(Equality_check)


if __name__ == "__main__":
    unittest.main()
