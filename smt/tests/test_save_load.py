import os
import unittest
import numpy as np
from smt.surrogate_models import KRG

class TestKRG(unittest.TestCase):
    
    def test_save_load(self):
        
        filename = "kriging_save_test"
        
        xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

        sm = KRG()
        sm.set_training_values(xt, yt)
        sm.train()

        sm._save(filename)
        self.assertTrue(os.path.exists(filename), f"Le fichier {filename} n'a pas été créé.")

        file_size = os.path.getsize(filename)
        print(f"Taille du fichier : {file_size} octets")

        sm2 = sm._load(filename)
        self.assertIsNotNone(sm2)

        num = 100
        x = np.linspace(0.0, 4.0, num).reshape(-1, 1)

        y = sm2.predict_values(x)

        self.assertIsNotNone(y)
        print("Prédictions avec le modèle chargé :", y)

        os.remove(filename)

if __name__ == "__main__":
    unittest.main()