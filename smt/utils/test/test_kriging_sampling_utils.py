"""
Author: Paul Saves
"""

import unittest
import numpy as np
from smt.surrogate_models import KRG
from smt.utils.krg_sampling import (
    covariance_matrix,
    sample_trajectory,
    gauss_legendre_grid,
    rectangular_grid,
    simpson_grid,
    eig_grid,
    sample_eigen,
)


class Test(unittest.TestCase):
    def test_cov_matrix(self):
        f = lambda x: x**2 * np.sin(x)

        x_min, x_max = -10, 10
        X_doe = np.array([-8.5, -4.0, -3.0, -1.0, 4.0, 7.5])
        Y_doe = f(X_doe)

        gp = KRG(theta0=[1e-2])
        gp.set_training_values(X_doe, Y_doe)
        gp._train()

        cov_matrix = covariance_matrix(gp, np.array([[-2], [0], [2]]), conditioned=True)

        self.assertAlmostEqual(cov_matrix.shape, (3, 3))

    def test_matrix_decomposition(self):
        f = lambda x: x**2 * np.sin(x)

        x_min, x_max = -10, 10
        X_doe = np.array([-8.5, -4.0, -3.0, -1.0, 4.0, 7.5])
        Y_doe = f(X_doe)

        gp = KRG(theta0=[1e-2])
        gp.set_training_values(X_doe, Y_doe)
        gp._train()

        n_plot = 20
        n_traj = 10
        X_data = np.linspace(x_min, x_max, n_plot).reshape(-1, 1)

        traj_chk = sample_trajectory(gp, X_data, n_traj, method="cholesky")
        traj_eig = sample_trajectory(gp, X_data, n_traj, method="eigen")

        self.assertEqual(traj_chk.shape, (n_plot, n_traj))
        self.assertEqual(traj_eig.shape, (n_plot, n_traj))

    def test_nystrom(self):
        f = lambda x: x**2 * np.sin(x)

        x_min, x_max = -10, 10
        X_doe = np.array([-8.5, -4.0, -3.0, -1.0, 4.0, 7.5])
        Y_doe = f(X_doe)
        bounds = np.array([[x_min], [x_max]])

        gp = KRG(theta0=[1e-2])
        gp.set_training_values(X_doe, Y_doe)
        gp._train()

        n_points = 10
        n_plot = 500
        n_traj = 20

        X_data = np.linspace(x_min, x_max, n_plot).reshape(-1, 1)

        x_grid_gl, weights_grid_gl = gauss_legendre_grid(bounds, n_points)
        x_grid_rec, weights_grid_rec = rectangular_grid(bounds, n_points)
        x_grid_sim, weights_grid_sim = simpson_grid(bounds, n_points)

        self.assertEqual(x_grid_gl.shape, (n_points, 1))
        self.assertEqual(x_grid_rec.shape, (n_points, 1))
        self.assertEqual(x_grid_sim.shape, (n_points, 1))

        self.assertEqual(weights_grid_gl.shape, (n_points, 1))
        self.assertEqual(weights_grid_rec.shape, (n_points, 1))
        self.assertEqual(weights_grid_sim.shape, (n_points, 1))

        eig_val_gl, eig_vec_gl, M_gl = eig_grid(gp, x_grid_gl, weights_grid_gl)
        eig_val_rec, eig_vec_rec, M_rec = eig_grid(gp, x_grid_rec, weights_grid_rec)
        eig_val_sim, eig_vec_sim, M_sim = eig_grid(gp, x_grid_sim, weights_grid_sim)

        self.assertEqual(eig_val_gl.shape, (n_points,))
        self.assertEqual(eig_val_rec.shape, (n_points,))
        self.assertEqual(eig_val_sim.shape, (n_points,))

        self.assertEqual(eig_vec_gl.shape, (n_points, n_points))
        self.assertEqual(eig_vec_rec.shape, (n_points, n_points))
        self.assertEqual(eig_vec_sim.shape, (n_points, n_points))

        self.assertEqual(M_gl, 9)
        self.assertEqual(M_rec, 9)
        self.assertEqual(M_sim, 9)

        traj_gl = sample_eigen(
            gp, X_data, eig_val_gl, eig_vec_gl, x_grid_gl, weights_grid_gl, M_gl, n_traj
        )
        traj_rec = sample_eigen(
            gp,
            X_data,
            eig_val_rec,
            eig_vec_rec,
            x_grid_rec,
            weights_grid_rec,
            M_rec,
            n_traj,
        )
        traj_sim = sample_eigen(
            gp,
            X_data,
            eig_val_sim,
            eig_vec_sim,
            x_grid_sim,
            weights_grid_sim,
            M_sim,
            n_traj,
        )

        self.assertEqual(traj_gl.shape, (n_plot, n_traj))
        self.assertEqual(traj_rec.shape, (n_plot, n_traj))
        self.assertEqual(traj_sim.shape, (n_plot, n_traj))


if __name__ == "__main__":
    unittest.main()
