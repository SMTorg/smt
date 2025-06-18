"""
Author: Hugo Reimeringer <hugo.reimeringer@onera.fr>
"""

import numpy as np
import scipy.optimize as opt
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd

from smt.applications.application import SurrogateBasedApplication
from smt.surrogate_models import KPLS, KPLSK, KRG
from smt.utils.checks import ensure_2d_array

PODI_available_models = {
    "KRG": KRG,
    "KPLS": KPLS,
    "KPLSK": KPLSK,
}


class SubspacesInterpolation:
    """
    Class that computes an interpolation of POD bases.

    Attributes
    ----------
    n_DoE : int
        Size of the DoE (Design of Experiments)
    n_mu : int
        dimension of the parametric space
    mu_train : np.ndarray[n_DoE, n_mu]
        Database of parameter values where the POD bases have been computed.
    n_DoF : int
        number of degree of freedom
    n_modes : int
        number of modes of the POD bases
    bases : np.ndarray[n_DoF, n_modes, n_DoE]
        Database of the POD bases to interpolate.
    print_global : bool
        Indicates if the indications plot (except the warnings) should be displayed.
    """

    def __init__(
        self, DoE_mu, DoE_bases, print_global, snapshots=None, U_ref=None
    ) -> None:
        """
        Initialization.

        Parameters
        ----------
        DoE_mu : np.ndarray[n_DoE, n_mu]
            Database of parameter values where the POD bases have been computed.
        DoE_bases : np.ndarray[n_DoF, n_modes, n_DoE]
            Database of the POD bases to interpolate.
        """
        self.print_global = print_global
        self.mu_train = DoE_mu
        self.bases = DoE_bases
        self.n_DoE = DoE_mu.shape[0]
        self.n_mu = DoE_mu.shape[1]
        self.n_DoF = DoE_bases.shape[0]
        self.n_modes = DoE_bases.shape[1]
        self.snapshots = snapshots
        if self.snapshots is not None:
            self.n_t = self.snapshots.shape[1]
        self.U_ref = U_ref

    def exponential_mapping(self, Y0, Ti) -> np.ndarray:
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
        Yi : np.ndarray [n_DoF,n_modes]
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

    def logarithmic_mapping(self, Y0, Yi) -> np.ndarray:
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

    def log_DoE(self, Y0) -> np.ndarray:
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
            Use optimal step size.

        Returns
        -------
        P_star: np.ndarray[n_DoF, n_modes]
            Computed Frechet mean
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

        u, s, _ = randomized_svd(Z_centered, n_components=self.n_DoE, random_state=0)
        # Information about the truncature
        self.n_B = np.argwhere(s.cumsum() / s.sum() >= 1 - epsilon)[0, 0] + 1
        if self.print_global:
            print(
                "The Grassmann manifold of interest is of dimension "
                + str((self.n_DoF - self.n_modes) * self.n_modes)
            )
            print("The number of tangent vectors is " + str(self.n_DoE))

            print(
                "The dimension of the subspace of the tangent plane is " + str(self.n_B)
            )
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
            self.compute_GP(self.alpha)
        return self.Basis, self.alpha, self.Z_mean

    def compute_GP(self, alpha, kernel="matern52") -> None:
        """
        Function that computes the GP interpolation functions of each alpha coefficients.

        Parameters
        ----------
        alpha: np.ndarray[n_B,n_modes]
            Set of generalized coefficients expressing the tangent vectors in a subspace of the tangent plane.
        kernel: str
            Choice of the kernel of the GPs see SMT documentation.
        """
        self.GP = []
        n = self.n_B
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
        coeff: np.ndarray[n_B, n_new]
            Mean value of the prediction.
        var: np.ndarray[n_B, n_new]
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
        Basis: np.ndarray[n_mu, n_real, n_DoF, n_modes]
            POD basis at points mu.
        Basis_real: np.ndarray[n_mu, n_real, n_DoF, n_modes]
            Random POD bases at points mu.
        """
        n_new = mu.shape[0]
        if not (compute_realizations):
            # compute the interpolation of the coefficients
            coeff = self.pred_coeff(mu)
            # compute the corresponding tangent vectors
            Zi = np.dot(self.Basis, coeff) + self.Z_mean[:, np.newaxis]
            # injectivity test to
            pca = PCA(svd_solver="randomized", random_state=42)
            pca.fit(Zi)
            singular_values = pca.singular_values_
            keep_injectivity = max(singular_values) < np.pi / 2
            if not (keep_injectivity):
                print(
                    "Warning : There is loss of injectivity for the interpolation of subspaces."
                )
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
                    f"Be sure to get sufficient memory for {n_real * n_new * self.n_DoF * self.n_modes} floats"
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


class PODI(SurrogateBasedApplication):
    """
    Class for Proper Orthogonal Decomposition and Interpolation (PODI) surrogate models based.

    Attributes
    ----------
    pod_type : str
        Indicates which type of POD should be performed ('global' or 'local')
    nx : int
        Dimension of the inputs in the DoE;
    n_snapshot : int
        Number of snapshots in the database.
    ny : int
        Dimension of the vector associated to a snapshot
    database : np.ndarray[ny, n_snapshot]
        Database containing the vectorial snapshots.
    n_modes : int
        Number of kept modes during the POD.
    basis : np.ndarray[ny, n_modes]
        POD basis.
    EV_ratio : float
        Ratio of explained variance according to the kept modes during the POD (only for global POD).
    singular_vectors : np.ndarray
        Singular vectors of the POD (only for global POD).
    singular_values : np.ndarray
        Singular values of the POD (only for global POD).
    interp_coeff : list[SurrogateModel]
        List containing the surrogate models used during the interpolation.

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
        self.basis = None

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

    @staticmethod
    def interp_subspaces(
        xt1,
        input_matrices,
        xn1,
        print_global=True,
        ref_index=0,
        frechet=False,
        frechet_guess=None,
        compute_realizations=False,
        n_realizations=1,
    ) -> list:
        """
        Static method computing the interpolation of subspaces.

        Parameters
        ----------
        xt1 : np.ndarray[nt1, dim]
            The input scalar values corresponding to each local bases.
            dim = dimension of parametric space for local bases
        input_bases : list[np.ndarray[ny, n_modes]]
            List containing the local bases for the subspaces' interpolation.
            Each matrix is associated to a scalar value from xt1.
        xn1 : np.ndarray[nn, dim]
            The scalar values at which we want to compute a local basis.
            dim = dimension of parametric space for local bases
        ref_index : int
            Index of the local base from 'input_matrices' we want to use as the reference point. Default value is 0.
            (Only if frechet set to False)
        frechet : bool
            Indicates if the frechet's mean should be computed.
        frechet_guess : np.ndarray[ny, n_modes]
            Initial guess for the frechet's mean. Default value is the fisrt local basis of 'input_matrices'.
            (Only if frechet set to True)
        compute_realizations : bool
            Indicates if some realizations of random POD bases should be computed.
        n_realizations : int
            In case compute_realizations is set to True, indicates the number of realizations that should be computed.

        Returns
        -------
        bases : list[np.ndarray[ny, n_modes]]
            List of the output bases at each desired value of xn1.
        bases_realizations : list[list[np.ndarray[ny, n_modes]]]
            List of realizations of the output bases. (Random POD bases)
            Returned only if compute_realizations is set to True.
            The list contain for each value of xn1 a list of the associated realizations.

        Examples
        --------
        #normal case
        >>> bases = PODI.interp_subspaces(xt1=xt1, input_bases=matrices, xn1=xn1)
        #use of frechet
        >>> bases = PODI.interp_subspaces(
                            xt1=xt1, input_bases=matrices, xn1=xn1,
                            frechet=True,
                            frechet_guess=frechet_matrix
                            )
        """
        nt1 = xt1.shape[0]
        n_bases = len(input_matrices)

        if nt1 != n_bases:
            raise ValueError(
                f"the xt1 rows should correspond to each basis, there are {nt1} xt1 rows, but {n_bases} input bases."
            )

        nn = xn1.shape[0]
        ny, n_modes = input_matrices[0].shape

        DoE_bases = np.zeros((ny, n_modes, n_bases))
        ########vérifier qu'en sortie matrice bonnes dimensions, pareil pour réalisations
        first_dim = input_matrices[0].shape
        for i, basis in enumerate(input_matrices):
            if basis.shape != first_dim:
                raise ValueError("The input matrices should have the same dimensions.")
            DoE_bases[:, :, i] = basis

        interp = SubspacesInterpolation(
            DoE_mu=xt1, DoE_bases=DoE_bases, print_global=print_global
        )

        if frechet:
            if frechet_guess is not None:
                P0 = frechet_guess
            else:
                P0 = input_matrices[0]
            Y0_frechet, _ = interp.compute_Frechet_mean(P0=P0)
            Y0 = Y0_frechet
        else:
            Y0 = input_matrices[ref_index]

        interp.compute_tangent_plane_basis_and_DoE_coordinates(Y0=Y0)
        if compute_realizations:
            Yi_full, Yi_full2 = interp.interp_POD_basis(
                xn1,
                compute_realizations=True,
                n_real=n_realizations,
            )
            yi = Yi_full
            interpolated_bases = []

            yi_real = []
            for real in range(Yi_full2.shape[1]):
                yi_real.append(Yi_full2[:, real, :, :])
            interpolated_bases_real = []

            for i in range(nn):
                interpolated_basis = yi[i, :, :]
                interpolated_bases.append(interpolated_basis)

                realizations_i = []
                for j in range(n_realizations):
                    realization_j_of_i = yi_real[j][i, :, :]
                    realizations_i.append(realization_j_of_i)
                interpolated_bases_real.append(realizations_i)

            return interpolated_bases, interpolated_bases_real
        else:
            Yi_full = interp.interp_POD_basis(xn1, compute_realizations=False)
            yi = Yi_full[:, 0, :, :]
            interpolated_bases = []
            for i in range(nn):
                interpolated_basis = yi[i, :, :]
                interpolated_bases.append(interpolated_basis)

            return interpolated_bases

    def compute_global_pod(
        self,
        tol: float = None,
        n_modes: int = None,
        seed: int = None,
    ) -> None:
        """
        Performs the global POD.

        Parameters
        ----------
        tol : float
            Desired tolerance for the pod (if n_modes not set).
        n_modes : int
            Desired number of kept modes for the pod (if tol not set).
        seed : int
            seed number which controls random draws for internal optim. (optional)
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
                "pod needs atleast one of the arguments 'n_modes' or 'tol'"
            )

        svd.fit(self.database.T)
        self.singular_vectors = svd.components_.T
        self.singular_values = svd.singular_values_
        EV_list = svd.explained_variance_

        if choice_svd == "tol":
            self.n_modes = PODI.choice_n_modes_tol(EV_list, tol)
        else:
            if n_modes > self.n_snapshot:
                raise ValueError(
                    "the number of kept modes can't be superior to the number of data values (snapshots)"
                )
        self.EV_ratio = sum(EV_list[: self.n_modes]) / sum(EV_list)
        self.EV_list = EV_list
        self.basis = self.singular_vectors[:, : self.n_modes]

    def compute_pod(
        self,
        database: np.ndarray,
        pod_type: str = "global",
        tol: float = None,
        n_modes: int = None,
        seed: int = None,
        local_basis: np.ndarray = None,
    ) -> None:
        """
        Performs the POD (global or local).

        Parameters
        ----------
        database : np.ndarray[ny, n_snapshot]
            Snapshot matrix. Each column corresponds to a snapshot.
        pod_type : str
            Name of the pod type that should be performed : 'global' or 'local'. Default value is 'global'.
        tol : float
            Desired tolerance for the pod (if n_modes not set).
            Only for global POD, pod_type set to 'global'.
        n_modes : int
            Desired number of kept modes for the pod (if tol not set).
            Only for global POD, pod_type set to 'global'.
        seed : int
            Seed number which controls random draws for internal optim. (optional)
            Only for global POD, pod_type set to 'global'.
        local_basis : np.ndarray[ny, n_modes]
            Local basis used for the local POD.

        Examples
        --------
        #global POD
        >>> sm.compute_pod(database, pod_type = 'global', tol = 0.99)
        #local POD
        >>> sm.compute_pod(database, pod_type = 'local', local_basis = basis)
        """
        self.database = ensure_2d_array(database, "database")
        self.n_snapshot = database.shape[1]

        self.ny = database.shape[0]

        self.mean = np.atleast_2d(database.mean(axis=1)).T

        if pod_type == "global":
            self.compute_global_pod(tol=tol, n_modes=n_modes, seed=seed)
        elif pod_type == "local":
            if local_basis is None:
                raise ValueError("'local_basis' should be specified")
            self.n_modes = local_basis.shape[1]

            ny = local_basis.shape[0]
            if ny != self.ny:
                raise ValueError(
                    f"the first dimension of the database and the local basis must be the same, {ny} != {self.ny}."
                )

            self.basis = local_basis
        else:
            raise ValueError(
                f"the pod type should be 'global' or 'local', not {pod_type}."
            )

        self.coeff = np.dot(database.T - self.mean.T, self.basis)

        self.pod_type = pod_type

        self.pod_computed = True
        self.interp_options_set = False

    @staticmethod
    def compute_pod_errors(
        xt: np.ndarray,
        database: np.ndarray,
        interp_type: str = "KRG",
        interp_options: list = [{}],
    ) -> list:
        """
        Calculates different errors for the POD.

        Parameters
        ----------
        xt : np.ndarray[n_snapshot, nx]
            The input values for the n_snapshot training points.

        database : np.ndarray[ny, n_snapshot]
            Snapshot matrix. Each column corresponds to a snapshot.

        interp_type : str
            Name of the type of surrogate model that will be used for the whole set.
            By default, the Kriging model is used (KRG).

        interp_options : list[dict]
            List containing dictionnaries for the options.
            The k-th dictionnary corresponds to the options of the k-th interpolation model.
            If the options are common to all surogate models, only a single dictionnary is required in the list.
            The available options can be found in the documentation of the corresponding surrogate models.
            By default, the print_global options are set to 'False'.

        Returns
        -------
        error_list : list[float]
            List of 3 POD errors : projection error, interpolation error and total error (projection and interpolation).
        """
        xt = ensure_2d_array(xt, "xt")

        nt = xt.shape[0]
        n_snapshot = database.shape[1]

        if nt != n_snapshot:
            raise ValueError(
                f"there must be the same amount of train values than data values, {nt} != {n_snapshot}."
            )

        max_interp_error = 0
        max_proj_error = 0
        max_total_error = 0

        for n in range(n_snapshot):
            reduced_database = np.concatenate(
                (database[:, :n], database[:, n + 1 :]), axis=1
            )
            reduced_xt = np.concatenate((xt[:n], xt[n + 1 :]))
            single_snapshot = np.atleast_2d(database[:, n]).T
            single_xt = np.atleast_2d(xt[n])

            podi = PODI()
            podi.compute_pod(
                database=reduced_database, n_modes=min(reduced_database.shape)
            )
            reduced_mean = np.atleast_2d(reduced_database.mean(axis=1)).T
            reduced_basis = podi.get_singular_vectors()
            n_modes = podi.get_n_modes()

            true_coeff = np.atleast_2d(
                np.dot(single_snapshot.T - reduced_mean.T, reduced_basis)
            ).T
            recomposed = reduced_mean + reduced_basis.dot(true_coeff)
            proj_error = recomposed - single_snapshot
            rms_proj_error = np.sqrt(np.mean(proj_error**2))
            max_proj_error = max(max_proj_error, rms_proj_error)

            podi.set_interp_options(
                interp_type=interp_type, interp_options=interp_options
            )
            podi.set_training_values(xt=reduced_xt)
            podi.train()
            reduced_interp_coeff = podi.get_interp_coeff()
            mean_coeff_interp = np.zeros((n_modes, 1))

            for i, coeff in enumerate(reduced_interp_coeff):
                mu_i = coeff.predict_values(single_xt)
                mean_coeff_interp[i] = mu_i[:, 0]

            interp_error = mean_coeff_interp - true_coeff
            rms_interp_error = np.sqrt(np.mean(interp_error**2))
            max_interp_error = max(max_interp_error, rms_interp_error)

            recomposed = reduced_mean + reduced_basis.dot(mean_coeff_interp)
            total_error = recomposed - single_snapshot
            rms_total_error = np.sqrt(np.mean(total_error**2))
            max_total_error = max(max_total_error, rms_total_error)

        return [max_interp_error, max_proj_error, max_total_error]

    def get_singular_vectors(self) -> np.ndarray:
        """
        Getter for the singular vectors of the global POD.
        It represents the directions of maximum variance in the data.

        Returns
        -------
        singular_vectors : np.ndarray
            Singular vectors of the global POD.
        """
        return self.singular_vectors

    def get_basis(self) -> np.ndarray:
        """
        Getter for the basis used for the POD.

        Returns
        -------
        basis : np.ndarray
            Basis of the POD.
        """
        return self.basis

    def get_singular_values(self) -> np.ndarray:
        """
        Getter for the singular values from the Sigma matrix of the POD.

        Returns
        -------
        singular_values : np.ndarray
            Singular values of the POD.
        """
        return self.singular_values

    def get_ev_list(self) -> float:
        """
        Getter for the explained variance list.

        Returns
        -------
        EV_ratio : float
            Explained variance ratio with the current kept modes.
        """
        return self.EV_list

    def get_ev_ratio(self) -> float:
        """
        Getter for the explained variance ratio with the kept modes.

        Returns
        -------
        EV_ratio : float
            Explained variance ratio with the current kept modes.
        """
        return self.EV_ratio

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

    def set_training_values(self, xt: np.ndarray) -> None:
        """
        Set training data (values).
        If the models' options are still not set, default values are used for the initialization.

        Parameters
        ----------
        xt : np.ndarray[n_snapshot, nx]
            The input values for the n_snapshot training points.
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
        nt = xt.shape[0]
        self.nx = xt.shape[1]
        if nt != self.n_snapshot:
            raise ValueError(
                f"there must be the same amount of train values than snapshots, {nt} != {self.n_snapshot}."
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
        xn : np.ndarray[n_new, nx]
            Input values for the prediction points.

        Returns
        -------
        yn : np.ndarray[n_new, nx]
            Output values at the prediction points.
        """
        if not self.train_done:
            raise RuntimeError(
                "the model should have been trained before trying to make a prediction."
            )

        xn = ensure_2d_array(xn, "xn")

        dim_new = xn.shape[1]

        if dim_new != self.nx:
            raise ValueError(
                f"the data values and the new values must be the same size, here {dim_new} != {self.nx}"
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
        xn : np.ndarray[n_new, nx]
            Input values for the prediction points.

        Returns
        -------
        s2 : np.ndarray[ny, n_new]
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
                f"the data values and the new values must be the same size, here {dim_new} != {self.nx}"
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
        xn : np.ndarray[n_new, nx]
            Input values for the prediction points.
        kx : int
            The 0-based index of the input variable with respect to which derivative is desired.

        Returns
        -------
        dy_dx : np.ndarray[ny, n_new]
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
                f"the data values and the new values must be the same size, here {dim_new} != {self.nx}"
            )

        self.n_new = xn.shape[0]
        deriv_coeff_interp = np.zeros((self.n_modes, self.n_new))

        for i in range(self.n_modes):
            deriv_coeff_interp[i] = self.interp_coeff[i].predict_derivatives(xn, d)[
                :, 0
            ]

        dy_dx = np.dot(self.basis, deriv_coeff_interp)

        return dy_dx

    def predict_variance_derivatives(self, xn, kx) -> np.ndarray:
        """
        Predict the derivatives of the variances at a set of points.

        Parameters
        ----------
        xn : np.ndarray[n_new, nx]
            Input values for the prediction points.
        kx : int
            The 0-based index of the input variable with respect to which derivative is desired.

        Returns
        -------
        dv_dx : np.ndarray[ny, n_new]
            Derivatives of the variances.
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
                f"the data values and the new values must be the same size, here {dim_new} != {self.nx}"
            )

        self.n_new = xn.shape[0]
        deriv_coeff_interp = np.zeros((self.n_modes, self.n_new))

        for i in range(self.n_modes):
            deriv_coeff_interp[i] = self.interp_coeff[i].predict_variance_derivatives(
                xn, d
            )[:, 0]

        dv_dx = np.dot(self.basis, deriv_coeff_interp)

        return dv_dx
