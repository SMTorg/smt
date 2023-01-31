"""
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.
"""

import matplotlib

matplotlib.use("Agg")

import unittest
import numpy as np
from sys import argv

from smt.applications import MOE, MOESurrogateModel
from smt.utils.sm_test_case import SMTestCase
from smt.problems import Branin, LpNorm
from smt.sampling_methods import FullFactorial, LHS
from smt.utils.misc import compute_rms_error
from smt.surrogate_models import RMTB, RMTC


class TestMOE(SMTestCase):
    """
    Test class
    """

    plot = None

    @staticmethod
    def function_test_1d(x):
        x = np.reshape(x, (-1,))
        y = np.zeros(x.shape)
        y[x < 0.4] = x[x < 0.4] ** 2
        y[(x >= 0.4) & (x < 0.8)] = 3 * x[(x >= 0.4) & (x < 0.8)] + 1
        y[x >= 0.8] = np.sin(10 * x[x >= 0.8])
        return y.reshape((-1, 1))

    # @unittest.skip('disabled')
    def test_1d_50(self):
        self.ndim = 1
        self.nt = 50
        self.ne = 50

        np.random.seed(0)
        xt = np.random.sample(self.nt).reshape((-1, 1))
        yt = self.function_test_1d(xt)
        moe = MOE(
            smooth_recombination=True,
            heaviside_optimization=True,
            n_clusters=3,
            xt=xt,
            yt=yt,
        )
        moe.train()

        # validation data
        np.random.seed(1)
        xe = np.random.sample(self.ne)
        ye = self.function_test_1d(xe)

        rms_error = compute_rms_error(moe, xe, ye)
        self.assert_error(rms_error, 0.0, 3e-1)

        self.assertRaises(RuntimeError, lambda: moe.predict_variances(xe))

        if TestMOE.plot:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            y = moe.predict_values(xe)
            plt.figure(1)
            plt.plot(ye, ye, "-.")
            plt.plot(ye, y, ".")
            plt.xlabel(r"$y$ actual")
            plt.ylabel(r"$y$ prediction")
            plt.figure(2)
            xv = np.linspace(0, 1, 100)
            yv = self.function_test_1d(xv)
            plt.plot(xv, yv, "-.")
            plt.plot(xe, y, "o")
            plt.show()

    # @unittest.skip('disabled')
    def test_1d_50_var(self):
        self.ndim = 1
        self.nt = 50
        self.ne = 50

        np.random.seed(0)
        xt = np.random.sample(self.nt).reshape((-1, 1))
        yt = self.function_test_1d(xt)
        moe = MOE(
            smooth_recombination=True,
            heaviside_optimization=True,
            n_clusters=3,
            xt=xt,
            yt=yt,
            variances_support=True,
        )
        moe.train()

        # validation data
        np.random.seed(1)
        xe = np.random.sample(self.ne)
        ye = self.function_test_1d(xe)

        rms_error = compute_rms_error(moe, xe, ye)
        self.assert_error(rms_error, 0.0, 3e-1)

        moe.predict_variances(xe)

        moe_hard = MOE(
            smooth_recombination=False,
            heaviside_optimization=True,
            n_clusters=3,
            xt=xt,
            yt=yt,
            variances_support=True,
        )
        moe_hard.train()
        moe_hard.predict_variances(xe)

        if TestMOE.plot:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            y = moe.predict_values(xe)
            plt.figure(1)
            plt.plot(ye, ye, "-.")
            plt.plot(ye, y, ".")
            plt.xlabel(r"$y$ actual")
            plt.ylabel(r"$y$ prediction")

            plt.figure(2)
            xv = np.linspace(0, 1, 100)
            yv = self.function_test_1d(xv)
            y = moe.predict_values(xv)
            y_std = np.sqrt(moe.predict_variances(xv))
            plt.plot(xv, yv, "--k", linewidth=1)
            plt.plot(xv, y, "-b", linewidth=1)
            plt.plot(xv, y + y_std, "--b", linewidth=1)
            plt.plot(xv, y - y_std, "--b", linewidth=1)
            plt.show()

    # @unittest.skip('disabled')
    def test_1d_50_surrogate_model(self):
        self.ndim = 1
        self.nt = 50
        self.ne = 50

        np.random.seed(0)
        xt = np.random.sample(self.nt).reshape((-1, 1))
        yt = self.function_test_1d(xt)
        moe = MOESurrogateModel(
            smooth_recombination=True,
            heaviside_optimization=True,
            n_clusters=3,
            xt=xt,
            yt=yt,
        )
        self.assertIsInstance(moe.moe, MOE)

        moe.train()
        self.assertFalse(moe.supports["variances"])

        # validation data
        np.random.seed(1)
        xe = np.random.sample(self.ne)
        ye = self.function_test_1d(xe)

        rms_error = compute_rms_error(moe, xe, ye)
        self.assert_error(rms_error, 0.0, 3e-1)

        self.assertRaises(RuntimeError, lambda: moe.predict_variances(xe))

        moe_var = MOESurrogateModel(
            smooth_recombination=True,
            heaviside_optimization=True,
            n_clusters=3,
            xt=xt,
            yt=yt,
            variances_support=True,
        )
        moe_var.train()
        self.assertTrue(moe_var.supports["variances"])
        moe_var.predict_variances(xe)

        if TestMOE.plot:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            y = moe.predict_values(xe)
            plt.figure(1)
            plt.plot(ye, ye, "-.")
            plt.plot(ye, y, ".")
            plt.xlabel(r"$y$ actual")
            plt.ylabel(r"$y$ prediction")
            plt.figure(2)
            xv = np.linspace(0, 1, 100)
            yv = self.function_test_1d(xv)
            plt.plot(xv, yv, "-.")
            plt.plot(xe, y, "o")
            plt.show()

    # @unittest.skip('disabled')
    def test_norm1_2d_200(self):
        self.ndim = 2
        self.nt = 200
        self.ne = 200

        prob = LpNorm(ndim=self.ndim)

        # training data
        sampling = FullFactorial(xlimits=prob.xlimits, clip=True)
        np.random.seed(0)
        xt = sampling(self.nt)
        yt = prob(xt)

        # mixture of experts
        moe = MOE(smooth_recombination=False, n_clusters=5)
        moe.set_training_values(xt, yt)
        moe.train()

        # validation data
        np.random.seed(1)
        xe = sampling(self.ne)
        ye = prob(xe)

        rms_error = compute_rms_error(moe, xe, ye)
        self.assert_error(rms_error, 0.0, 1e-1)

        if TestMOE.plot:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            y = moe.predict_values(xe)
            plt.figure(1)
            plt.plot(ye, ye, "-.")
            plt.plot(ye, y, ".")
            plt.xlabel(r"$y$ actual")
            plt.ylabel(r"$y$ prediction")

            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(xt[:, 0], xt[:, 1], yt)
            plt.title("L1 Norm")
            plt.show()

    # @unittest.skip('disabled for now as it blocks unexpectedly on travis linux')
    def test_branin_2d_200(self):
        self.ndim = 2
        self.nt = 200
        self.ne = 200

        prob = Branin(ndim=self.ndim)

        # training data
        sampling = FullFactorial(xlimits=prob.xlimits, clip=True)
        np.random.seed(0)
        xt = sampling(self.nt)
        yt = prob(xt)

        # mixture of experts
        moe = MOE(n_clusters=5)
        moe.set_training_values(xt, yt)
        moe.options["heaviside_optimization"] = True
        moe.train()

        # validation data
        np.random.seed(1)
        xe = sampling(self.ne)
        ye = prob(xe)

        rms_error = compute_rms_error(moe, xe, ye)
        self.assert_error(rms_error, 0.0, 1e-1)

        if TestMOE.plot:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            y = moe.analyse_results(x=xe, operation="predict_values")
            plt.figure(1)
            plt.plot(ye, ye, "-.")
            plt.plot(ye, y, ".")
            plt.xlabel(r"$y$ actual")
            plt.ylabel(r"$y$ prediction")

            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(xt[:, 0], xt[:, 1], yt)
            plt.title("Branin function")
            plt.show()

    def test_enabled_expert_types(self):
        moe = MOE(variances_support=True)
        expected = ["KPLS", "KPLSK", "KRG"]
        self.assertEqual(expected, sorted(moe.enabled_experts))

        moe = MOE(derivatives_support=True)
        expected = ["IDW", "KPLS", "KPLSK", "KRG", "LS", "QP", "RBF", "RMTB", "RMTC"]
        self.assertEqual(expected, sorted(moe.enabled_experts))

        moe = MOE(deny=["KRG", "RMTB"])
        expected = ["IDW", "KPLS", "KPLSK", "LS", "QP", "RBF", "RMTC"]
        self.assertEqual(expected, sorted(moe.enabled_experts))

        moe = MOE(allow=["KRG", "RMTB"])
        expected = ["KRG", "RMTB"]
        self.assertEqual(expected, sorted(moe.enabled_experts))

        moe = MOE(variances_support=True, allow=["KRG", "RMTB"])
        expected = ["KRG"]
        self.assertEqual(expected, sorted(moe.enabled_experts))

        moe = MOE(derivatives_support=True, deny=["RBF", "IDW", "KPLSK"])
        expected = ["KPLS", "KRG", "LS", "QP", "RMTB", "RMTC"]
        self.assertEqual(expected, sorted(moe.enabled_experts))

    def test_fix_moe_rmts_bug(self):
        def myfunc(x):
            return -0.5 * (
                np.sin(40 * (x - 0.85) ** 4) * np.cos(2.5 * (x - 0.95))
                + 0.5 * (x - 0.9)
                + 1
            )

        nt1 = 11
        nt2 = 15
        ne = 101

        # Training data
        X1 = np.linspace(0.001, 0.3, nt1).reshape(nt1, 1)
        X1 = np.concatenate((X1, np.array([[0.35]])), axis=0)
        X2 = np.linspace(0.4, 1.0, nt2).reshape(nt2, 1)
        xt = np.concatenate((X1, X2), axis=0)
        yt = myfunc(xt)

        moe = MOE(smooth_recombination=True, n_clusters=2, heaviside_optimization=True)
        moe._surrogate_type = {"RMTB": RMTB, "RMTC": RMTC}
        moe.set_training_values(xt, yt)
        moe.train()

    def test_fix_test_data_bug(self):
        # MOE does not interpolate the first training point
        def myfunc(x):
            return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)

        # limits of teh design space
        xlimits = np.array([[0.0, 1.0]])
        # LHS DOE with ndoe points
        ndoe = 6

        # Construction of the DOE
        sampling = LHS(xlimits=xlimits, criterion="m", random_state=0)
        x1D = sampling(ndoe)
        x1D = np.sort(x1D, axis=0)
        # Compute the output
        y1D = myfunc(x1D)

        # test data
        num = 50
        xv1D = sampling(num)
        xv1D = np.sort(xv1D, axis=0)
        yv1D = myfunc(xv1D)

        moe1D = MOE(n_clusters=1, xtest=xv1D, ytest=yv1D, allow=["KRG"])
        moe1D.set_training_values(x1D, y1D)
        moe1D.train()

        # Check that moe1D is interpolating all training values
        ypred = moe1D.predict_values(x1D)
        self.assertTrue(np.allclose(y1D, ypred))

    def test_bad_allow_value(self):
        nt = 35
        sampling = FullFactorial(xlimits=np.array([[0, 1]]), clip=True)
        np.random.seed(0)
        xt = sampling(nt)
        yt = self.function_test_1d(xt)

        moe = MOE(n_clusters=1, allow=["TOTO"])
        moe.set_training_values(xt, yt)
        with self.assertRaises(ValueError) as context:
            moe.train()
        self.assertEqual(
            "List of experts is empty: check support, allow and deny options wrt "
            "possible experts: ['KRG', 'KPLS', 'KPLSK', 'LS', 'QP', 'RBF', 'IDW', 'RMTB', 'RMTC']",
            str(context.exception),
        )

    @staticmethod
    def run_moe_example_1d():
        import numpy as np
        from smt.applications import MOE
        from smt.sampling_methods import FullFactorial
        import matplotlib.pyplot as plt

        nt = 35

        def function_test_1d(x):
            import numpy as np  # Note: only required by SMT doc testing toolchain

            x = np.reshape(x, (-1,))
            y = np.zeros(x.shape)
            y[x < 0.4] = x[x < 0.4] ** 2
            y[(x >= 0.4) & (x < 0.8)] = 3 * x[(x >= 0.4) & (x < 0.8)] + 1
            y[x >= 0.8] = np.sin(10 * x[x >= 0.8])
            return y.reshape((-1, 1))

        x = np.linspace(0, 1, 100)
        ytrue = function_test_1d(x)

        # Training data
        sampling = FullFactorial(xlimits=np.array([[0, 1]]), clip=True)
        np.random.seed(0)
        xt = sampling(nt)
        yt = function_test_1d(xt)

        # Mixture of experts
        print("MOE Experts: ", MOE.AVAILABLE_EXPERTS)

        # MOE1: Find the best surrogate model on the whole domain
        moe1 = MOE(n_clusters=1)
        print("MOE1 enabled experts: ", moe1.enabled_experts)
        moe1.set_training_values(xt, yt)
        moe1.train()
        y_moe1 = moe1.predict_values(x)

        # MOE2: Set nb of cluster with just KRG, LS and IDW surrogate models
        moe2 = MOE(smooth_recombination=False, n_clusters=3, allow=["KRG", "LS", "IDW"])
        print("MOE2 enabled experts: ", moe2.enabled_experts)
        moe2.set_training_values(xt, yt)
        moe2.train()
        y_moe2 = moe2.predict_values(x)

        fig, axs = plt.subplots(1)
        axs.plot(x, ytrue, ".", color="black")
        axs.plot(x, y_moe1)
        axs.plot(x, y_moe2)
        axs.set_xlabel("x")
        axs.set_ylabel("y")
        axs.legend(["Training data", "MOE 1 Prediction", "MOE 2 Prediction"])

        plt.show()

    @staticmethod
    def run_moe_example_2d():
        import numpy as np
        from smt.applications import MOE
        from smt.problems import LpNorm
        from smt.sampling_methods import FullFactorial

        import sklearn
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from mpl_toolkits.mplot3d import Axes3D

        ndim = 2
        nt = 200
        ne = 200

        # Problem: L1 norm (dimension 2)
        prob = LpNorm(ndim=ndim)

        # Training data
        sampling = FullFactorial(xlimits=prob.xlimits, clip=True)
        np.random.seed(0)
        xt = sampling(nt)
        yt = prob(xt)

        # Mixture of experts
        print("MOE Experts: ", MOE.AVAILABLE_EXPERTS)

        moe = MOE(smooth_recombination=True, n_clusters=5, deny=["RMTB", "KPLSK"])
        print("Enabled Experts: ", moe.enabled_experts)
        moe.set_training_values(xt, yt)
        moe.train()

        # Validation data
        np.random.seed(1)
        xe = sampling(ne)
        ye = prob(xe)

        # Prediction
        y = moe.predict_values(xe)
        fig = plt.figure(1)
        fig.set_size_inches(12, 11)

        # Cluster display
        colors_ = list(colors.cnames.items())
        GMM = moe.cluster
        weight = GMM.weights_
        mean = GMM.means_
        if sklearn.__version__ < "0.20.0":
            cov = GMM.covars_
        else:
            cov = GMM.covariances_
        prob_ = moe._proba_cluster(xt)
        sort = np.apply_along_axis(np.argmax, 1, prob_)

        xlim = prob.xlimits
        x0 = np.linspace(xlim[0, 0], xlim[0, 1], 20)
        x1 = np.linspace(xlim[1, 0], xlim[1, 1], 20)
        xv, yv = np.meshgrid(x0, x1)
        x = np.array(list(zip(xv.reshape((-1,)), yv.reshape((-1,)))))
        prob = moe._proba_cluster(x)

        plt.subplot(221, projection="3d")
        ax = plt.gca()
        for i in range(len(sort)):
            color = colors_[int(((len(colors_) - 1) / sort.max()) * sort[i])][0]
            ax.scatter(xt[i][0], xt[i][1], yt[i], c=color)
        plt.title("Clustered Samples")

        plt.subplot(222, projection="3d")
        ax = plt.gca()
        for i in range(len(weight)):
            color = colors_[int(((len(colors_) - 1) / len(weight)) * i)][0]
            ax.plot_trisurf(
                x[:, 0], x[:, 1], prob[:, i], alpha=0.4, linewidth=0, color=color
            )
        plt.title("Membership Probabilities")

        plt.subplot(223)
        for i in range(len(weight)):
            color = colors_[int(((len(colors_) - 1) / len(weight)) * i)][0]
            plt.tricontour(x[:, 0], x[:, 1], prob[:, i], 1, colors=color, linewidths=3)
        plt.title("Cluster Map")

        plt.subplot(224)
        plt.plot(ye, ye, "-.")
        plt.plot(ye, y, ".")
        plt.xlabel("actual")
        plt.ylabel("prediction")
        plt.title("Predicted vs Actual")

        plt.show()


if __name__ == "__main__":
    if "--plot" in argv:
        TestMOE.plot = True
        argv.remove("--plot")
    if "--example" in argv:
        TestMOE.run_moe_example_1d()
        exit()
    unittest.main()
