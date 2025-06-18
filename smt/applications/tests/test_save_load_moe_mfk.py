import os
import unittest

import numpy as np

from smt.applications.mfk import MFK, NestedLHS
from smt.applications.mfkpls import MFKPLS
from smt.applications.mfkplsk import MFKPLSK
from smt.applications.moe import MOE
from smt.sampling_methods import FullFactorial


class TestSaveLoad(unittest.TestCase):
    def function_test_1d(self, x):
        x = np.reshape(x, (-1,))
        y = np.zeros(x.shape)
        y[x < 0.4] = x[x < 0.4] ** 2
        y[(x >= 0.4) & (x < 0.8)] = 3 * x[(x >= 0.4) & (x < 0.8)] + 1
        y[x >= 0.8] = np.sin(10 * x[x >= 0.8])
        return y.reshape((-1, 1))

    def test_save_load_moe(self):
        filename = "moe_save_test"
        nt = 35
        x = np.linspace(0, 1, 100)

        sampling = FullFactorial(xlimits=np.array([[0, 1]]), clip=True)
        np.random.seed(0)
        xt = sampling(nt)
        yt = self.function_test_1d(xt)

        moe1 = MOE(n_clusters=1)
        moe1.set_training_values(xt, yt)
        moe1.train()
        y_moe1 = moe1.predict_values(x)

        moe1.save(filename)

        moe2 = MOE.load(filename)
        y_moe2 = moe2.predict_values(x)

        np.testing.assert_allclose(y_moe1, y_moe2)

        os.remove(filename)

    def lf_function(self, x):
        return 0.5 * ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2) + (x - 0.5) * 10.0 - 5

    def hf_function(self, x):
        return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)

    def _setup_MFKs(self):
        xlimits = np.array([[0.0, 1.0]])
        xdoes = NestedLHS(nlevel=2, xlimits=xlimits, random_state=0)
        xt_c, xt_e = xdoes(7)
        yt_e = self.hf_function(xt_e)
        yt_c = self.lf_function(xt_c)
        x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)
        return (xt_c, xt_e, yt_c, yt_e, x)

    def test_save_load_mfk_mfkpls_mfkplsk(self):
        filename = "MFKs_save_test"
        MFKs = [MFK, MFKPLS, MFKPLSK]
        xt_c, xt_e, yt_c, yt_e, x = self._setup_MFKs()
        ncomp = 1
        for mfk in MFKs:
            if mfk == MFK:
                application = MFK(theta0=xt_e.shape[1] * [1.0], corr="squar_exp")
            elif mfk == MFKPLS:
                application = MFKPLS(n_comp=ncomp, theta0=ncomp * [1.0])
            else:
                application = MFKPLSK(n_comp=ncomp, theta0=ncomp * [1.0])

            application.set_training_values(xt_c, yt_c, name=0)
            application.set_training_values(xt_e, yt_e)

            application.train()
            application.save(filename)

            x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)
            y1 = application.predict_values(x)

            mfk2 = MFK.load(filename)
            y2 = mfk2.predict_values(x)

            np.testing.assert_allclose(y1, y2)

            os.remove(filename)


if __name__ == "__main__":
    unittest.main()
