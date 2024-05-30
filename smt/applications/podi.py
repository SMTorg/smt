"""
Author: Hugo Reimeringer <hugo.reimeringer@onera.fr>
"""

import warnings

import scipy.optimize as opt
from sklearn.utils.extmath import randomized_svd

from sklearn.decomposition import PCA
import numpy as np

from sklearn.model_selection import train_test_split

from smt.applications.application import SurrogateBasedApplication
from smt.surrogate_models import KRG, KPLS, KPLSK
from smt.utils.checks import ensure_2d_array

PODI_available_models = {
    "KRG": KRG,
    "KPLS": KPLS,
    "KPLSK": KPLSK,
}

class MatrixInterpolation():
    """
    Class that computes a GP interpolation of POD bases.
    
    Construction:\n
    Inputs:
    --------------------------------------------
    DoE_mu : np.ndarray[n_DoE, n_mu]
        Database of parameter values where the POD bases have been computed.
        n_DoE = DoE size, n_mu = dimension of the parametric space
    DoE_bases : np.ndarray[n_DoF,n_modes,n_DoE]
        Database of the POD bases to interpolate.
        n_DoF = number of degree of freedom, n_modes = number of modes of the POD bases, n_DoE = DoE size
    """
    
    def __init__(self, DoE_mu, DoE_bases, snapshots=None, U_ref=None):
        self.mu_train = DoE_mu
        self.bases = DoE_bases
        self.n_DoE = DoE_mu.shape[0]
        self.n_mu = DoE_mu.shape[1]
        self.n_DoF = DoE_bases.shape[0]
        self.n_modes = DoE_bases.shape[1]
        self.snapshots = snapshots
        if type(self.snapshots) != type(None):
            self.n_t = self.snapshots.shape[1]
        self.U_ref = U_ref
    
    def exponential_mapping(self, Y0, Ti):
        """
        Function that computes the exponential mapping of the Grassmann manifold at point Y0.
    
        Parameters
        ----------
        Y0 : np.ndarray[n_DoF, n_modes]
            A point on the Grassmann manifold Gr(n_DoF,n_modes).
        Ti : np.ndarray[n_DoF, n_modes]
            A vector in the tangent plane of Gr(n_DoF,n_modes) at Y0.
    
        Returns
        -------
        Yi : np.ndarray [..,n_DoF,n_modes]
            A point on Gr(n_DoF,n_modes).
            The image of Ti by the exponential mapping at Y0. $$Yi=exp_{Y0}(Ti)$$
        """
        u, s, v = np.linalg.svd(Ti, full_matrices=False)
        n_new = u.shape[0]
        n_real = u.shape[1]
        M = np.repeat(np.eye(u.shape[3])[:, :, np.newaxis], n_real, axis=2)
        M = np.repeat(M[:, :, :, np.newaxis], n_new, axis=3)
        coss = M * np.cos(s).T
        coss = np.transpose(coss, (3, 2, 0, 1))
        sins = M * np.sin(s).T
        sins = np.transpose(sins, (3, 2, 0, 1))
        vt = np.transpose(v, (0, 1, 3, 2))
        Yi = Y0 @ (vt @ coss) + u @ sins
        return Yi
    
    def logarithmic_mapping(self, Y0, Yi):
        """
        Function that computes the logarithmic mapping of the Grassmann manifold at point Y0.
    
        Parameters
        ----------
        Y0: np.ndarray[n_DoF,n_modes]
            A point on the Grassmann manifold Gr(n_DoF,n_modes)
        Yi: np.ndarray[n_DoF,n_modes]
            A point on the Grassmann manifold Gr(n_DoF,n_modes)
    
        Returns
        -------
        Ti: np.ndarray[n_DoF,n_modes]
            A vector on the tangent plane of Gr(n_DoF,n_modes) at Y0.
            The image of Yi by the logarithmic mapping at Y0. $$Ti=log_{Y0}(Yi)$$
        """
        X = np.linalg.solve(np.dot(Y0.T, Yi), np.eye(Y0.shape[1]))
        M = np.dot(Yi, X) - Y0
        u, s, v = np.linalg.svd(M, full_matrices=False)
        arctans = np.eye(Y0.shape[1]) * np.arctan(s)
        Ti = np.dot(u, np.dot(arctans, v))
        return Ti
    
    def log_DoE(self, Y0):
        """
        Function that computes the logarithmic mapping of the DoE at a given point Y0.
    
        Parameters
        ----------
        Y0: np.ndarray[n_DoF,n_modes]
            A point on the Grassmann manifold Gr(n_DoF,n_modes).
    
        Returns
        -------
        T_train: np.ndarray[n_DoF,n_modes,n_DoE]
            n_DoE vectors on Gr(n_DoF,n_modes), images of the DoE by the logarithmic mapping at Y0
        """
        T_train = np.zeros((self.n_DoF, self.n_modes, self.n_DoE))
        for i in range(self.n_DoE):
            T_train[:, :, i] = self.logarithmic_mapping(Y0, self.bases[:, :, i])
        return T_train
    
    def compute_Frechet_mean(
        self, P0, itermax=20, epsilon=1e-6, alpha_0=1e-3, optimal_step=True
    ):
        """
        Function that computes the Frechet mean of the set DoE_bases.
    
        Parameters
        ----------
        P0: np.ndarray[n_DoF,n_modes]
            Initial guess for the Frechet mean.
        iter_max: int, default = 20
            Maximum number of iterations in the gradient descend.
        epsilon: float, defaut = 1e-6
            Threshold for the stopping criterion.
        alpha_0: float,  default = 1e-3
            Initial guess for the step of the gradient descend algorithm.
        optimal_step: bool, default = True
            Use optimal step size .
        """
        it = 0
        cv = 1
        if optimal_step:
            # define the function that will be optimized to find the optimal step size
            def obj_fun_i(alpha, P_i, delta_P):
                Ti = alpha * delta_P
                P_i_temp = self.exponential_mapping(
                    P_i[np.newaxis, np.newaxis, :, :], Ti[np.newaxis, np.newaxis, :, :]
                )
                delta_P_temp = self.log_DoE(P_i_temp[0, 0])
                delta_P_temp_norm = np.linalg.norm(delta_P_temp, axis=(0, 1)) ** 2
                return delta_P_temp_norm.sum()
    
        i = 0
        obj_fun = []
        while it < itermax and cv > epsilon:
            # compute the gradient
            delta_P = self.log_DoE(P0)
            obj_fun.append(np.linalg.norm(delta_P, axis=(0, 1)) ** 2)
            delta_P = delta_P.sum(axis=2)
            if optimal_step:
                res = opt.minimize(obj_fun_i, alpha_0, args=(P0, delta_P))
                step = res["x"]
            else:
                step = alpha_0
    
            delta_P = step * delta_P
            P_i = self.exponential_mapping(
                P0[np.newaxis, np.newaxis, :, :], delta_P[np.newaxis, np.newaxis, :, :]
            )[0, 0]
            if i > 0:
                cv = np.linalg.norm(obj_fun[i] - obj_fun[i - 1]) / np.linalg.norm(
                    obj_fun[i]
                )
            P0 = P_i
            i += 1
            it += 1
        P_star = P_i
        return P_star, obj_fun
    
    def compute_tangent_plane_basis_and_DoE_coordinates(
        self, Y0, epsilon=1e-3, compute_GP=True
    ):
        """
        Function that computes a basis of a subspace of the tangent plane at Y0
        It is done from a set of vector that belong to this tangent plane,
        and the coefficients of the tangent vectors in this basis.
    
        Parameters
        ----------
        Y0: np.ndarray[n_DoF,n_modes]
            A point on the Grassmann manifold Gr(n_DoF,n_modes).
        epsilon: float
            Truncation parameter, the basis is created by keeping the n_B first modes such that
            $$\frac{\\sum_i=1^{n_B} s_i}{\\sum_i=1^{n_DoE} s_i}>1-epsilon
        compute_GP: bool (default=True)
            Computes the GP approximation of each coefficient
    
        Returns
        -------
        Basis: np.ndarray[n_DoF*n_modes,n_B]
            Basis of the subspace of the tangent plane of interest.
        alpha: np.ndarray[n_B,n_modes]
            Generalized coefficients of the tangent vectors in the new basis.
        Z_mean: np.ndarray[n_DoF*n_modes,]
            Mean value of the flatten tangent vectors.
        """
        self.Y0 = Y0
        # compute the vectors on the tangent plane at Y0
        T_train = self.log_DoE(Y0)
        # flatten the tangent vectors
        Z = np.zeros((self.n_DoF * self.n_modes, self.n_DoE))
        for i in range(self.n_DoE):
            Z[:, i] = T_train[:, :, i].flatten()
        self.Z_mean = Z.mean(axis=1)
        Z_centered = (Z.T - self.Z_mean).T
        # svd of the Z matrix:
    
        u, s, v = randomized_svd(Z_centered, n_components=self.n_DoE, random_state=0)
        # Information about the truncature
        print(
            "The Grassmann manifold of interest is of dimension "
            + str((self.n_DoF - self.n_modes) * self.n_modes)
        )
        print("The number of tangent vectors is " + str(self.n_DoE))
        self.n_B = np.argwhere(s.cumsum() / s.sum() >= 1 - epsilon)[0, 0] + 1
        print("The dimension of the subspace of the tangent plane is " + str(self.n_B))
        if self.n_B == self.n_DoE:
            print(
                "WARNING: the dimension of the tangent plane's subspace is equal to the DoE size."
            )
            print("Consider increasing the DoE.")
    
        # truncature
        self.Basis = u[:, : self.n_B]
    
        # projection to get the coefficients in this basis
        self.alpha = np.dot(self.Basis.T, Z_centered)
        if compute_GP:
            GP = self.compute_GP(self.alpha)
        return self.Basis, self.alpha, self.Z_mean
    
    def compute_GP(self, alpha, kernel="matern52"):
        """
        Function that computes the GP interpolation functions of each alpha coefficients.
    
        Parameters
        ----------
        alpha: np.ndarray[n_B,n_modes]
            Set of generalized coefficients expressing the tangent vectors in a subspace of the tangent plane.
        kernel: str
            Choice of the kernel of the GPs see SMT documentation.
    
        Returns
        -------
        GPs: list
            List of SMT object GP, one per coefficients.
        """
        self.GP = []
        n=self.n_B
        for i in range(n):
            gp = KRG(
                theta0=[1e-2] * self.n_mu,
                print_prediction=False,
                corr=kernel,
                print_global=False,
            )
            gp.set_training_values(self.mu_train, alpha[i, :])
            gp.train()
            self.GP.append(gp)
        return self.GP
    
    def pred_coeff(self, mu, compute_var=False):
        """
        Function that interpolates the coefficients of the tangent vector at n_new parametric points mu.
    
        Parameters
        ----------
        mu: np.ndarray[n_new,n_mu]
            Parametric points where the interpolation should be provided.
        compute_var: bool (default=False)
            Computes the variance of the interpolation.
    
        Returns
        -------
        coeff: np.ndarray[n_B,n_new]
            Mean value of the prediction.
        var: np.ndarray[n_B,n_new]
            Variance of the prediction.
        """
        n_new = mu.shape[0]
        n = self.n_B

        coeff = np.zeros((n, n_new))
        for i in range(n):
            coeff[i] = self.GP[i].predict_values(mu)[:, 0]
        if not (compute_var):
            return coeff
        elif compute_var:
            var = np.zeros((n, n_new))
            for i in range(n):
                var[i] = self.GP[i].predict_variances(mu)[:, 0]
            return coeff, var
    
    def interp_POD_basis(
        self, mu, compute_realizations=False, n_real=1, fixed_xi=False, xi=None
    ):
        """
        Function that computes the interpolation of the POD basis at n_new parametric points mu.
    
        Parameters
        ----------
        mu: np.ndarray[n_new,n_mu]
            Parametric points where the interpolation should be provided.
        realizations: bool (default=False)
            Compute n_real random realizations of the interpolation.
        n_real: float
            Number of random realizations.
    
        Returns
        -------
        Basis: np.ndarray[n_mu,n_real,n_DoF,n_modes]
            POD basis at points mu.
        Basis_real: np.ndarray[n_mu,n_real,n_DoF,n_modes]
            Random POD bases at points mu.
        """
        n_new = mu.shape[0]
        if not (compute_realizations):
            # compute the interpolation of the coefficients
            coeff = self.pred_coeff(mu)
            # compute the corresponding tangent vectors
            Zi = np.dot(self.Basis, coeff) + self.Z_mean[:, np.newaxis]
            # reshape to get the matrices
            Ti = Zi.reshape((self.n_DoF, self.n_modes, n_new, 1))
            # transpose to apply the SVD in exp map to each matrix
            Ti = np.transpose(Ti, (2, 3, 0, 1))
            ind = abs(Ti) < 1e-12
            Ti[ind] = 0.0
            # exponential mapping
            Yi = self.exponential_mapping(self.Y0, Ti)
    
            return Yi
        elif compute_realizations:
            if n_new != 1:
                print(
                    f"WARNING: Sample of size {n_real} at {n_new} points of a {self.n_DoF}*{self.n_modes} matrix"
                )
                print(
                    f"Be sure to get sufficient memory for {n_real*n_new*self.n_DoF*self.n_modes} floats"
                )
            # compute the interpolation of the coefficients and the associated variance
            coeff, var = self.pred_coeff(mu, compute_var=True)
            # get a normal random vector
            if not (fixed_xi):
                xi = np.random.normal(size=(len(coeff), n_real))
            else:
                xi = xi
            # we append a zeros vector so that the last realization is the MAP
            xi = np.concatenate((xi, np.zeros((len(coeff), 1))), axis=1)
            n_real = n_real + 1
            var_coeff = np.sqrt(var).T[:, :, np.newaxis] * xi
            coeff_MC = (coeff.T)[:, :, np.newaxis] + var_coeff
            # compute the tangent vector
            Zi = np.dot(self.Basis, coeff_MC) + self.Z_mean[:, np.newaxis, np.newaxis]
            # reshape to get the matrix
            Ti = Zi.reshape((self.n_DoF, self.n_modes, n_new, n_real))
            # transpose to apply the SVD in exp map to each matrix
            Ti = np.transpose(Ti, (2, 3, 0, 1))
            # exponential mapping
            Yi = self.exponential_mapping(self.Y0, Ti)
            return Yi[:, -1, :, :], Yi[:, :-1, :, :]
    
    def GP_GC(self, Yi, separate_variables=True, DoE=None):
        """
        Function that computes the GP interpolation of the generalized coordinates (GC) on a given basis.
    
        Parameters
        ----------
        Yi: np.ndarray[n_DOF,n_modes]
    
        separate_variables: bool (default=True)
            Either to interpolate the GC by a separation of variables approach or not.
        DoE: np.ndarray
            If separate_variables is set to False, DoE to use to learn the GP of the GC.
    
        Returns
        -------
        GP_GC: list
            List of the GP objects that interpolate each GC
            If separate_variables is set to True, each list contains n GP for each GC.
        Bases_GC: list
            If separate_variables is set to True, list of basis for the reconstruction of the GC.
        Mean_GC: list
            If separate_variables is set to True, list of mean for the reconstruction of the GC.
        """
        # project the snapshots on Yi to get a DoE of GC
        Yi = Yi[np.newaxis, :, :]
        Yi_t = np.transpose(Yi, (0, 2, 1))
        snapshots_t = np.transpose(self.snapshots, (0, 2, 1))
        GC_DoE = Yi_t @ snapshots_t
        if separate_variables:
            self.GP_coeff = []
            self.Bases_coeff = []
            self.Mean_coeff = []
            for i in range(self.n_modes):
                beta_i = GC_DoE[:, i, :]
                mean_i = beta_i.mean(axis=0)
                self.Mean_coeff.append(mean_i)
                beta_i = beta_i - mean_i
                # we express beta_i as a linear combination of temporal modes
                u, s, v = np.linalg.svd(beta_i.T)
                B_i = u[:, : self.n_DoE]
                self.Bases_coeff.append(B_i)
                gamma_i = np.dot(B_i.T, beta_i.T)
                gp_coeff = []
                for j in range(self.n_DoE):
                    gp = KRG(
                        theta0=[1e-2] * (self.n_mu),
                        print_prediction=False,
                        corr="matern32",
                    )
                    gp.set_training_values(self.mu_train, gamma_i[j, :])
                    gp.train()
                    gp_coeff.append(gp)
                self.GP_coeff.append(gp_coeff)
    
            return self.GP_coeff, self.Bases_coeff, self.Mean_coeff
    
        elif not (separate_variables):
            self.GP_coeff = []
            for i in range(self.n_modes):
                beta_i = GC_DoE[:, i, :]
                gp = KRG(
                    theta0=[1e-2] * (DoE.shape[1]),
                    print_prediction=False,
                    corr="squar_exp",
                )
                gp.set_training_values(DoE, beta_i.flatten())
                gp.train()
                gp_coeff.append(gp)
                self.GP_coeff.append(gp_coeff)
            return self.GP_coeff
    
    def approx_NI_cst_basis(self, Yi, mu, realizations=False, n_real=None):
        """
        Computes the approximation of the response by non intrusive approach.
        Uncertainty quantification can be performed with respect to the GC interpolation.
    
        Parameters
        ----------
        Yi: np.ndarray[n_DOF,n_modes]
    
        mu: np.ndarray[1,n_mu]
            Parametric points where the interpolation should be provided.
        realizations: bool (default=False)
            Computes n_real random realizations of the interpolation.
        n_real: int
            If realizations is set to True, number of samples to draw.
    
        Returns
        -------
        Approx: np.ndarray
            Approximation of the response at mu on the ROB Yi.
        """
        if realizations:
            U_approx_MC = []
            gamma_hat = np.zeros((self.n_modes, self.n_DoE))
            gamma_hat_var = np.zeros((self.n_modes, self.n_DoE))
            for i in range(self.n_modes):
                for j in range(self.n_DoE):
                    gamma_hat[i, j] = self.GP_coeff[i][j].predict_values(mu)[0]
                    gamma_hat_var[i, j] = self.GP_coeff[i][j].predict_variances(mu)[0]
            for n in range(n_real):
                coeff_hat = np.zeros((self.n_modes, self.n_t))
                for i in range(self.n_modes):
                    gamma_hat_real = np.zeros((self.n_DoE,))
                    for j in range(self.n_DoE):
                        xi = np.random.normal()
                        gamma_hat_real[j] = (
                            gamma_hat[i, j] + np.sqrt(gamma_hat_var[i, j]) * xi
                        )
                    coeff_hat[i, :] = self.Mean_coeff[i] + np.dot(
                        self.Bases_coeff[i], gamma_hat_real
                    )
    
                U_approx_hat = np.dot(Yi, coeff_hat).T + self.U_ref
                U_approx_MC.append(U_approx_hat)
            U_approx_MC = np.array(U_approx_MC)
            return U_approx_MC
        elif not (realizations):
            gamma_hat = np.zeros((self.n_modes, self.n_DoE))
            for i in range(self.n_modes):
                for j in range(self.n_DoE):
                    gamma_hat[i, j] = self.GP_coeff[i][j].predict_values(mu)[0]
    
            coeff_hat = np.zeros((self.n_modes, self.n_t))
            for i in range(self.n_modes):
                gamma_hat_real = np.zeros((self.n_DoE,))
                for j in range(self.n_DoE):
                    gamma_hat_real[j] = gamma_hat[i, j]
                coeff_hat[i, :] = self.Mean_coeff[i] + np.dot(
                    self.Bases_coeff[i], gamma_hat_real
                )
    
            U_approx_hat = np.dot(Yi, coeff_hat).T + self.U_ref
    
            return U_approx_hat
    
    def approx_NI(
        self,
        mu,
        UQ_basis=False,
        UQ_GC=False,
        n_real_basis=None,
        n_real_GC=None,
        fixed_xi_basis=False,
        xi_basis=None,
    ):
        """
        Computes a GP interpolation of both the basis and the GC.
        Interpolation uncertainty for both the basis and te GC can be evaluated by Monte Carlo sampling.
    
        Parameters
        ----------
        mu: np.ndarray[1,n_mu]
            Parametric points where the interpolation should be provided.
        UQ_basis: bool (default=False)
            Weither to performe UQ for the POD basis interpolation.
        UQ_GC: bool (default=False)
            Weither to performe UQ for the GC interpolation.
        n_real_basis: int
            If UQ_basis is set to True, size of the Monte Carlo sample for the basis.
        n_real_GC: int
            If UQ_GC is set to True, size of the Monte Carlo sample for the GC.
    
        WARNING : if UQ_basis set to True and UQ_GC set to True, the MC sample size is n_real_basis*n_real_GC.
    
        Returns
        -------
        """
        if not (UQ_basis) and not (UQ_GC):
            Yi = self.interp_POD_basis(mu, compute_realizations=False, n_real=1)
            # res = self.GP_GC(Yi[0, 0])
            approx = self.approx_NI_cst_basis(Yi[0, 0], mu)
            return approx
        elif UQ_basis and not (UQ_GC):
            if not (fixed_xi_basis):
                Yi, Yi_real = self.interp_POD_basis(
                    mu, compute_realizations=True, n_real=n_real_basis
                )
            else:
                Yi, Yi_real = self.interp_POD_basis(
                    mu,
                    compute_realizations=True,
                    n_real=n_real_basis,
                    fixed_xi=fixed_xi_basis,
                    xi=xi_basis,
                )
            # res = self.GP_GC(Yi[0])
            approx = self.approx_NI_cst_basis(Yi[0], mu)
            approx_MC = []
            for i in range(n_real_basis):
                # res = self.GP_GC(Yi_real[0, i])
                approx_MC.append(self.approx_NI_cst_basis(Yi_real[0, i], mu))
            return approx, approx_MC
        elif UQ_basis and UQ_GC:
            if not (fixed_xi_basis):
                Yi, Yi_real = self.interp_POD_basis(
                    mu, compute_realizations=True, n_real=n_real_basis
                )
            else:
                Yi, Yi_real = self.interp_POD_basis(
                    mu,
                    compute_realizations=True,
                    n_real=n_real_basis,
                    fixed_xi=fixed_xi_basis,
                    xi=xi_basis,
                )
            # res = self.GP_GC(Yi[0])
            approx = self.approx_NI_cst_basis(Yi[0], mu)
            approx_MC = []
            for i in range(n_real_basis):
                # res = self.GP_GC(Yi_real[0, i])
                approx_MC.append(
                    self.approx_NI_cst_basis(
                        Yi_real[0, i], mu, realizations=True, n_real=n_real_GC
                    )
                )
            return approx, approx_MC
        elif not (UQ_basis) and UQ_GC:
            Yi = self.interp_POD_basis(mu, compute_realizations=False, n_real=1)
            # res = self.GP_GC(Yi[0, 0])
            approx = self.approx_NI_cst_basis(Yi[0, 0], mu)
            approx_MC = self.approx_NI_cst_basis(
                Yi[0, 0], mu, realizations=True, n_real=n_real_GC
            )
            return approx, approx_MC
    
    def interp_POD_basis_RBF(self, mu, Y0, epsilon=1.0):
        """
        Function that computes the interpolation of the POD basis at a new parametric point mu by RBF.
    
        Parameters
        ----------
        mu: np.ndarray[1,n_mu]
            Parametric points where the interpolation should be provided.
        Y0: np.ndarray[n_DoF,n_modes]
            A point on the Grassmann manifold Gr(n_DoF,n_modes).
        epsilon : float
            Correlation lenght of the gaussian kernel.
    
        Returns
        -------
        Basis: np.ndarray[n_DoF,n_modes]
            POD basis at points mu.
        """
        R1 = np.repeat(self.mu_train, self.n_DoE, 0)
        R2 = np.tile(self.mu_train, (self.n_DoE, 1))
        norm = np.linalg.norm(R1 - R2, axis=1)
        Phi = self._gaussian_kernel(norm, epsilon)
        Phi = Phi.reshape((self.n_DoE, self.n_DoE))
        norm_new = np.linalg.norm(self.mu_train - mu, axis=1)
        phi = self._gaussian_kernel(norm_new, epsilon)
        w = np.linalg.solve(Phi, phi)
        # DoE vetors on the tangent plane
        T_train = self.log_DoE(Y0)
        # interpolation
        T_new = np.sum(T_train * w, axis=2)
    
        # exponential mapping
        Yi = self.exponential_mapping(Y0, T_new[np.newaxis, np.newaxis, :, :])
        return Yi[0, 0]
    
    def _gaussian_kernel(self, r, epsilon=1.0):
        return np.exp(-((r / epsilon) ** 2))
    
    def interp_POD_basis_IDW(self, mu, P0, p=2, epsilon=1e-3, itermax=100):
        """
        Function that computes the interpolation of the POD basis at a new parametric point mu by IDW.
    
        Parameters
        -----------
        mu: np.ndarray[1,n_mu]
            Parametric points where the interpolation should be provided.
        P0: np.ndarray[n_DoF,n_modes]
            Initial guess.
        p: float
            The inverse distance is defined as a function of d(mu,mu_i)**p
    
        Returns
        -------
        Y_i: np.ndarray[n_DoF,n_modes]
            POD basis at points mu.
        l_res: list
            List containing the value of the norm of the gradient of the optimization problem.
        itermax: int, default=100
            Max number of iteration for the resolution of the optimization problem.
        """
        # Compute weights
        norm_new = np.linalg.norm(self.mu_train - mu, axis=1)
        ind = norm_new == 0.0
        if ind.any():
            alpha = np.zeros((len(ind),))
            alpha[ind] = 1.0
        else:
            S = np.sum(1 / norm_new**p)
            alpha = 1 / (S * norm_new**p)
        n_iter = 0
        l_res = []
        res = 1.0
        while res > epsilon and n_iter < itermax:
            T_train = self.log_DoE(P0)
            T_new = 1 / len(alpha) * np.sum(T_train * alpha, axis=2)
            res = np.linalg.norm(T_new)
            l_res.append(res)
            Yi = self.exponential_mapping(P0, T_new[np.newaxis, np.newaxis, :, :])[0, 0]
            P0 = Yi
            n_iter += 1
    
        return Yi, l_res, n_iter

class PODI(SurrogateBasedApplication):
    """
    Class for Proper Orthogonal Decomposition and Interpolation (PODI) surrogate models based.

    Attributes
    ----------
    pod_type : str
        Indicates which type of POD should be performed ('global' or 'local')
    n_modes : int
        Number of kept modes during the POD.
    singular_vectors : np.ndarray
        Singular vectors of the POD.
    singular_values : np.ndarray
        Singular values of the POD.
    interp_coeff : list[SurrogateModel]
        List containing the surrogate models used.

    Examples
    --------
    >>> from smt.applications import PODI
    >>> sm = PODI()
    """

    name = "PODI"

    def _initialize(self) -> None:
        super(PODI, self)._initialize()

        self.available_models_name = []
        for key in PODI_available_models.keys():
            self.available_models_name.append(key)

        self.pod_type = None
        self.n_modes = None
        self.singular_vectors = None
        self.singular_values = None

        self.pod_computed = False
        self.interp_options_set = False
        self.training_values_set = False
        self.train_done = False

        self.interp_coeff = None

    @staticmethod
    def choice_n_modes_tol(EV_list: np.ndarray, tol: float) -> int:
        """
        Static method calculating the required number of kept modes to explain at least the intended ratio of variance.

        Parameters
        ----------
        EV_list : np.ndarray
            List of the explained variance of each mode in the POD.
        tol : float
            Desired tolerance for the POD. It is expected to be in ]0, 1].

        Returns
        -------
        n_modes : int
            Kept modes according to the tolerance.
        """

        sum_tot = sum(EV_list)
        sum_ = 0
        for i, ev in enumerate(EV_list):
            sum_ += ev
            EV_ratio = sum_ / sum_tot
            if EV_ratio >= tol:
                return i + 1
        return len(EV_list)

    def compute_global_pod(
        self,
        database: np.ndarray,
        tol: float = None,
        n_modes: int = None,
        compute_proj_error = True,
        seed: int = None
    ) -> None:
        ############compute erreur projection et interpolation et tout ce qu'on veut
        """
        Performs the global POD.

        Parameters
        ----------
        database : np.ndarray[ny, nt]
            Snapshot matrix. Each column corresponds to a snapshot.
        tol : float
            Desired tolerance for the pod (if n_modes not set).
        n_modes : int
            Desired number of kept modes for the pod (if tol not set).
        seed : int
            seed number which controls random draws for internal optim. (optional)

        Examples
        ----------
        >>> sm.compute_pod(database, tol = 0.99)
        """
        choice_svd = None

        svd = PCA(svd_solver="randomized", random_state=seed)

        if n_modes is not None:
            self.n_modes = n_modes
            choice_svd = "mode"

        if tol is not None:
            if choice_svd is not None:
                raise ValueError(
                    "pod can't use both arguments 'n_modes' and 'tol' at the same time"
                )
            else:
                choice_svd = "tol"

        if choice_svd is None:
            raise ValueError(
                "either one of the arguments 'n_modes' and 'tol' must be specified"
            )

        svd.fit(database.T)
        self.singular_vectors = svd.components_.T
        self.singular_values = svd.singular_values_
        EV_list = svd.explained_variance_

        if choice_svd == "tol":
            self.n_modes = PODI.choice_n_modes_tol(EV_list, tol)
        else:
            if self.n_modes > self.n_snapshot:
                raise ValueError(
                    "the number of kept modes can't be superior to the number of data values (snapshots)"
                )
        self.EV_ratio = sum(EV_list[: self.n_modes]) / sum(EV_list)

        self.basis = self.singular_vectors[:, : self.n_modes]

        if compute_proj_error:
            true_coeff_list = []
            ceoff_list = []
            for n in range(self.n_snapshot):
                reducted_database = np.concatenate((database[:,:n], database[:,n+1:]), axis=1)
                svd.fit(reducted_database.T)
                basis = svd.components_.T
                coeff = np.dot(reducted_database.T - self.mean.T, basis)

                single_vector = database[:,n]
                true_coeff = np.dot(single_vector - self.mean.T, basis)
                true_coeff_list.append(coeff)
                sm = KRG(print_global = False)
                reducted_xt = np.concatenate((xt[:n], xt[n+1:]))
                single_input = xt[n]
                sm.set_training_values(reducted_xt, )

            




    @staticmethod
    def interp_subspaces(xt1, input_matrices, xn, frechet = False):
        ###############frechet paramètre
        ###############liste de matrices
        ###############méthode employée
        ############### pas de compute error projection
        ############### changer nom
        nn = xn.shape[0]

        ny, n_modes = input_matrices[0].shape
        ###nombre de mode doit être cohérent avec database, dimension ny aussi, nombre de bases cohérentavec nombre de paramètres
        n_bases = len(input_matrices)
        DoE_bases = np.zeros((ny, n_modes, n_bases))
        for i, basis in enumerate(input_matrices):
            DoE_bases[:,:,i] = basis

        interp = MatrixInterpolation(DoE_mu = xt1, DoE_bases = input_matrices)

        if frechet:
            Y0_frechet, _ = interp.compute_Frechet_mean(P0 = input_matrices[:,:,0])
            Y0 = Y0_frechet
        else:
            Y0 = input_matrices[:,:,0]
        interp.compute_tangent_plane_basis_and_DoE_coordinates(Y0 = Y0)
        Yi_full = interp.interp_POD_basis(xn)
        yi = np.squeeze(Yi_full, axis = 1)
        interpolated_bases = []
        for i in range(nn):
            interpolated_basis = yi[i,:,:]
            interpolated_bases.append(interpolated_basis)
        
        return interpolated_bases

    def compute_pod(
        self,
        database: np.ndarray,
        pod_type: str = "global",
        tol: float = None,
        n_modes: int = None,
        seed: int = None,
        compute_proj_error = False,
        xt = None,
        interpolated_basis: np.ndarray = None
    ) -> None:
        database = ensure_2d_array(database, "database")
        self.n_snapshot = database.shape[1]
        self.ny = database.shape[0]

        self.mean = np.atleast_2d(database.mean(axis=1)).T

        if pod_type == "global":
            self.compute_global_pod(
                database=database, tol=tol, n_modes=n_modes, compute_proj_error = compute_proj_error, seed=seed
            )
        elif pod_type == "local":
            if interpolated_basis == None:
                raise ValueError(
                    "'interpolated_basis' should be specified"
                )
            self.n_modes = interpolated_basis.shape[1]
            self.basis = interpolated_basis
        else:
            raise ValueError(
                f"the pod type should be 'global' or 'local', not {pod_type}."
            )

        self.coeff = np.dot(database.T - self.mean.T, self.basis)
        self.pod_type = pod_type

        self.pod_computed = True
        self.interp_options_set = False
        self.training_values_set = False
    
    @staticmethod
    def compute_projection_error(basis, test_ratio = 0.1, seed = 42, print_values = False):
        return None
            
    def get_singular_vectors(self) -> np.ndarray:
        """
        Getter for the singular vectors of the POD.
        It represents the directions of maximum variance in the data.

        Returns
        -------
        singular_vectors : np.ndarray
            singular vectors of the POD.
        """
        return self.singular_vectors  ###############only if global ?

    def get_singular_values(self) -> np.ndarray:
        """
        Getter for the singular values from the Sigma matrix of the POD.

        Returns
        -------
        singular_values : np.ndarray
            Singular values of the POD.
        """
        return self.singular_values  #############only if global ?

    def get_ev_ratio(self) -> float:
        """
        Getter for the explained variance ratio with the kept modes.

        Returns
        -------
        EV_ratio : float
            Explained variance ratio with the current kept modes.
        """
        return self.EV_ratio  ################## only if global ?

    def get_n_modes(self) -> int:
        """
        Getter for the number of modes kept during the POD.

        Returns
        -------
        n_modes : int
            number of modes kept during the POD.
        """
        return self.n_modes

    def set_interp_options(
        self, interp_type: str = "KRG", interp_options: list = [{}]
    ) -> None:
        """
        Set the options for the interpolation surrogate models used.
        Only required if a model different than KRG is used or if non-default options are desired for the models.

        Parameters
        ----------
        interp_type : str
            Name of the type of surrogate model that will be used for the whole set.
            By default, the Kriging model is used (KRG).

        interp_options : list[dict]
            List containing dictionnaries for the options.
            The k-th dictionnary corresponds to the options of the k-th interpolation model.
            If the options are common to all surogate models, only a single dictionnary is required in the list.
            The available options can be found in the documentation of the corresponding surrogate models.
            By default, the print_global options are set to 'False'.

        Examples
        --------
        >>> interp_type = "KRG"
        >>> dict1 = {'corr' : 'matern52', 'theta0' : [1e-2]}
        >>> dict2 = {'poly' : 'quadratic'}
        >>> interp_options = [dict1, dict2]
        >>> sm.set_interp_options(interp_type, interp_options)
        """

        if not self.pod_computed:
            raise RuntimeError(
                "'compute_pod' method must have been succesfully executed before trying to set the models options."
            )

        if interp_type not in PODI_available_models.keys():
            raise ValueError(
                f"the surrogate model type should be one of the following : {', '.join(self.available_models_name)}"
            )

        if len(interp_options) == self.n_modes:
            mode_options = "local"
        elif len(interp_options) == 1:
            mode_options = "global"
        else:
            raise ValueError(
                f"expected interp_options of size {self.n_modes} or 1, but got {len(interp_options)}."
            )

        self.interp_coeff = []
        for i in range(self.n_modes):
            if mode_options == "local":
                index = i
            elif mode_options == "global":
                index = 0

            sm_i = PODI_available_models[interp_type](print_global=False)

            for key in interp_options[index].keys():
                sm_i.options[key] = interp_options[index][key]

            self.interp_coeff.append(sm_i)

        self.interp_options_set = True
        self.training_values_set = False

    def set_training_values(self, xt: np.ndarray) -> None:
        """
        Set training data (values).
        If the models' options are still not set, default values are used for the initialization.

        Parameters
        ----------
        xt : np.ndarray[nt, nx]
            The input values for the nt training points.
        """

        xt = ensure_2d_array(xt, "xt")

        if not self.pod_computed:
            raise RuntimeError(
                "'compute_pod' method must have been succesfully executed before trying to set the training values."
            )
        if not self.interp_options_set:
            self.interp_coeff = []
            for i in range(self.n_modes):
                sm_i = PODI_available_models["KRG"](print_global=False)

                self.interp_coeff.append(sm_i)
            self.interp_options_set = True

        self.nt = xt.shape[0]
        self.nx = xt.shape[1]

        if self.nt != self.n_snapshot:
            raise ValueError(
                f"there must be the same amount of train values than data values, {self.nt} != {self.n_snapshot}."
            )
        for i in range(self.n_modes):
            self.interp_coeff[i].set_training_values(xt, self.coeff[:, i])

        self.training_values_set = True

    def train(self) -> None:
        """
        Performs the training of the model.
        """
        if not self.pod_computed:
            raise RuntimeError(
                "'compute_pod' method must have been succesfully executed before trying to train the models."
            )

        if not self.training_values_set:
            raise RuntimeError(
                "the training values should have been set before trying to train the models."
            )

        for interp_coeff in self.interp_coeff:
            interp_coeff.train()

        self.train_done = True

    def get_interp_coeff(self) -> np.ndarray:
        """
        Getter for the list of the interpolation surrogate models used

        Returns
        -------
        interp_coeff : np.ndarray[n_modes]
            List of the kriging models used for the POD coefficients.
        """
        return self.interp_coeff

    def predict_values(self, xn) -> np.ndarray:
        """
        Predict the output values at a set of points.

        Parameters
        ----------
        xn : np.ndarray
            Input values for the prediction points.

        Returns
        -------
        yn : np.ndarray
            Output values at the prediction points.
        """
        if not self.train_done:
            raise RuntimeError(
                "the model should have been trained before trying to make a prediction."
            )

        xn = ensure_2d_array(xn, "xn")

        self.dim_new = xn.shape[1]

        if self.dim_new != self.nx:
            raise ValueError(
                f"the data values and the new values must be the same size, here {self.dim_new} != {self.nx}"
            )

        self.n_new = xn.shape[0]
        mean_coeff_interp = np.zeros((self.n_modes, self.n_new))

        for i in range(self.n_modes):
            mu_i = self.interp_coeff[i].predict_values(xn)
            mean_coeff_interp[i] = mu_i[:, 0]

        y = self.mean + np.dot(self.basis, mean_coeff_interp)

        return y

    def predict_variances(self, xn) -> np.ndarray:
        """
        Predict the variances at a set of points.

        Parameters
        ----------
        xn : np.ndarray[nt, nx]
            Input values for the prediction points.

        Returns
        -------
        s2 : np.ndarray[nt, ny]
            Variances.
        """

        if not self.train_done:
            raise RuntimeError(
                "the model should have been trained before trying to make a prediction"
            )

        xn = ensure_2d_array(xn, "xn")

        dim_new = xn.shape[1]

        if dim_new != self.nx:
            raise ValueError(
                f"the data values and the new values must be the same size, here {self.dim_new} != {self.nx}"
            )

        self.n_new = xn.shape[0]
        var_coeff_interp = np.zeros((self.n_modes, self.n_new))

        for i in range(self.n_modes):
            sigma_i_square = self.interp_coeff[i].predict_variances(xn)
            var_coeff_interp[i] = sigma_i_square[:, 0]

        s2 = np.dot((self.basis**2), var_coeff_interp)

        return s2

    def predict_derivatives(self, xn, kx) -> np.ndarray:
        """
        Predict the dy_dx derivatives at a set of points.

        Parameters
        ----------
        xn : np.ndarray[nt, nx]
            Input values for the prediction points.
        kx : int
            The 0-based index of the input variable with respect to which derivative is desired.

        Returns
        -------
        dy_dx : np.ndarray[nt, ny]
            Derivatives.
        """
        d = kx

        if not self.train_done:
            raise RuntimeError(
                "the model should have been trained before trying to make a prediction"
            )

        xn = ensure_2d_array(xn, "xn")

        dim_new = xn.shape[1]

        if d >= dim_new:
            raise ValueError(
                "the desired derivative kx should correspond to a dimension of the data, here kx is out of bounds."
            )

        if dim_new != self.nx:
            raise ValueError(
                f"the data values and the new values must be the same size, here {self.dim_new} != {self.nx}"
            )

        self.n_new = xn.shape[0]
        deriv_coeff_interp = np.zeros((self.n_modes, self.n_new))

        for i in range(self.n_modes):
            deriv_coeff_interp[i] = self.interp_coeff[i].predict_derivatives(xn, d)[
                :, 0
            ]

        dy_dx = np.dot(self.basis, deriv_coeff_interp)

        return dy_dx
