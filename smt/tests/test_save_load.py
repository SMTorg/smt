import os
import unittest
import numpy as np

from smt.problems import Sphere
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG, LS, KPLS, GEKPLS, KPLSK, MGP, QP, SGP, GENN, QP


class TestSaveLoad(unittest.TestCase):
    
    def test_save_load_GEKPLS(self):
        
        filename = "sm_save_test"
        fun = Sphere(ndim=2)

        sampling = LHS(xlimits=fun.xlimits, criterion="m")
        xt = sampling(20)
        yt = fun(xt)

        for i in range(2):
            yd = fun(xt, kx=i)
            yt = np.concatenate((yt, yd), axis=1)

        sm = GEKPLS()
        sm.set_training_values(xt, yt[:, 0])
        for i in range(2):
            sm.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)
        sm.train()

        sm.save(filename)
        self.assertTrue(os.path.exists(filename), f"Le fichier {filename} n'a pas été créé.")

        file_size = os.path.getsize(filename)
        print(f"Taille du fichier : {file_size} octets")

        sm2 = GEKPLS.load(filename)
        self.assertIsNotNone(sm2)

        X = np.arange(fun.xlimits[0, 0], fun.xlimits[0, 1], 0.25)
        Y = np.arange(fun.xlimits[1, 0], fun.xlimits[1, 1], 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros((X.shape[0], X.shape[1]))

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = sm.predict_values(
                    np.hstack((X[i, j], Y[i, j])).reshape((1, 2))
                ).item()

        self.assertIsNotNone(Z)
        print("Prédictions avec le modèle chargé :", Z)

        os.remove(filename)

    def test_save_krigs(self):
        
        krigs = [KRG, KPLS, KPLSK, MGP, SGP, QP, GENN, LS]
        rng = np.random.RandomState(1)
        N_inducing = 30

        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

        for surrogate in krigs:
            
            filename = "sm_save_test"

            sm = surrogate()
            sm.set_training_values(xt, yt)

            if surrogate == SGP:
                sm.Z = 2 * rng.rand(N_inducing, 1) - 1
                sm.set_inducing_inputs(Z=sm.Z)

            sm.train()

            sm.save(filename)
            self.assertTrue(os.path.exists(filename), f"Le fichier {filename} n'a pas été créé.")

            file_size = os.path.getsize(filename)
            print(f"Taille du fichier : {file_size} octets")

            sm2 = surrogate.load(filename)
            self.assertIsNotNone(sm2)

            num = 100
            x = np.linspace(0.0, 4.0, num).reshape(-1, 1)

            y = sm2.predict_values(x)

            self.assertIsNotNone(y)
            print("Prédictions avec le modèle chargé :", y)

            os.remove(filename)

if __name__ == "__main__":
    unittest.main()