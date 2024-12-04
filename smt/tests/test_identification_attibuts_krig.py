import unittest
import numpy as np
from smt.surrogate_models import KRG

class TestIdKrig(unittest.TestCase):
    
    def test_attrib_krig(self):
        
        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

        sm = KRG()
        sm.set_training_values(xt, yt)
        sm.train()
        nx = sm.nx
        nt = sm.nt
        design_space = sm.design_space
        X_offset = sm.X_offset
        X_scale = sm.X_scale
        X_norma = sm.X_norma
        optimal_par = sm.optimal_par
        y_mean = sm.y_mean
        y_std = sm.y_std
        ny = sm.ny
        print(f"nx = ", nx)
        print(f"nt = ", nt)
        print(f"design space = ", design_space)

        self.assertIsNotNone(nx)

        sm2 = KRG()
        sm2.nx = nx
        sm2.nt = nt
        sm2.design_space = design_space
        sm2.X_offset = X_offset
        sm2.X_scale = X_scale
        sm2.X_norma = X_norma
        sm2.optimal_par = optimal_par
        sm2.y_mean = y_mean
        sm2.y_std = y_std
        sm2.ny = ny
        sm2.predict_values(np.array([5, 1]))
        print

if __name__ == "__main__":
    unittest.main()