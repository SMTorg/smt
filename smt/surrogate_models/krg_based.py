"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
Some functions are copied from gaussian_process submodule (Scikit-learn 0.14)
This package is distributed under New BSD license.
"""
import numpy as np
from scipy import linalg, optimize
from copy import deepcopy
import warnings

from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.kriging_utils import differences
from smt.utils.kriging_utils import constant, linear, quadratic
from smt.utils.kriging_utils import (
    squar_exp,
    abs_exp,
    act_exp,
    standardization,
    cross_distances,
    matern52,
    matern32,
    gower_componentwise_distances,
    componentwise_distance,
    componentwise_distance_PLS,
    compute_X_cont,
    cross_levels,
    compute_X_cross,
    cross_levels_homo_space,
)
from scipy.stats import multivariate_normal as m_norm
from smt.sampling_methods import LHS
from smt.utils.mixed_integer import unfold_with_enum_mask

# TODO: Create hyperclass Kernels and a class for each kernel
EXP_HOMO_HSPHERE_KERNEL = "exponential_homoscedastic_matrix_kernel"
HOMO_HSPHERE_KERNEL = "homoscedastic_matrix_kernel"
CONT_RELAX_KERNEL = "continuous_relaxation_matrix_kernel"
GOWER_KERNEL = "gower_matrix_kernel"


class KrgBased(SurrogateModel):

    _regression_types = {"constant": constant, "linear": linear, "quadratic": quadratic}

    _correlation_types = {
        "abs_exp": abs_exp,
        "squar_exp": squar_exp,
        "act_exp": act_exp,
        "matern52": matern52,
        "matern32": matern32,
    }

    name = "KrigingBased"

    def _initialize(self):
        super(KrgBased, self)._initialize()
        declare = self.options.declare
        supports = self.supports
        declare(
            "poly",
            "constant",
            values=("constant", "linear", "quadratic"),
            desc="Regression function type",
            types=(str),
        )
        declare(
            "corr",
            "squar_exp",
            values=(
                "abs_exp",
                "squar_exp",
                "act_exp",
                "matern52",
                "matern32",
            ),
            desc="Correlation function type",
        )
        declare(
            "categorical_kernel",
            None,
            values=[
                CONT_RELAX_KERNEL,
                GOWER_KERNEL,
                EXP_HOMO_HSPHERE_KERNEL,
                HOMO_HSPHERE_KERNEL,
            ],
            desc="The kernel to use for categorical inputs. Only for non continuous Kriging",
        )

        declare(
            "xtypes",
            None,
            types=list,
            desc="x type specifications: either FLOAT for continuous, INT for integer "
            "or (ENUM n) for categorical dimension with n levels",
        )
        declare(
            "nugget",
            100.0 * np.finfo(np.double).eps,
            types=(float),
            desc="a jitter for numerical stability",
        )
        declare(
            "theta0", [1e-2], types=(list, np.ndarray), desc="Initial hyperparameters"
        )
        # In practice, in 1D and for X in [0,1], theta^{-2} in [1e-2,infty), i.e.
        # theta in (0,1e1], is a good choice to avoid overfitting. By standardising
        # X in R, X_norm = (X-X_mean)/X_std, then X_norm in [-1,1] if considering
        # one std intervals. This leads to theta in (0,2e1]
        declare(
            "theta_bounds",
            [1e-6, 2e1],
            types=(list, np.ndarray),
            desc="bounds for hyperparameters",
        )
        declare(
            "hyper_opt",
            "Cobyla",
            values=("Cobyla", "TNC"),
            desc="Optimiser for hyperparameters optimisation",
            types=(str),
        )
        declare(
            "eval_noise",
            False,
            types=bool,
            values=(True, False),
            desc="noise evaluation flag",
        )
        declare(
            "noise0",
            [0.0],
            types=(list, np.ndarray),
            desc="Initial noise hyperparameters",
        )
        declare(
            "noise_bounds",
            [100.0 * np.finfo(np.double).eps, 1e10],
            types=(list, np.ndarray),
            desc="bounds for noise hyperparameters",
        )
        declare(
            "use_het_noise",
            False,
            types=bool,
            values=(True, False),
            desc="heteroscedastic noise evaluation flag",
        )
        declare(
            "n_start",
            10,
            types=(int),
            desc="number of optimizer runs (multistart method)",
        )
        self.best_iteration_fail = None
        self.nb_ill_matrix = 5
        supports["derivatives"] = True
        supports["variances"] = True
        supports["variance_derivatives"] = True

    def _new_train(self):
        # Sampling points X and y
        X = self.training_points[None][0][0]
        y = self.training_points[None][0][1]

        # Compute PLS-coefficients (attr of self) and modified X and y (if GEKPLS is used)
        if self.name not in ["Kriging", "MGP"]:
            if self.options["categorical_kernel"] is None:
                X, y = self._compute_pls(X.copy(), y.copy())

        self._check_param()
        self.X_train = X

        if self.options["categorical_kernel"] is not None:
            D, self.ij, X = gower_componentwise_distances(
                X=X, xtypes=self.options["xtypes"]
            )
            self.Lij, self.n_levels = cross_levels(
                X=self.X_train, ij=self.ij, xtypes=self.options["xtypes"]
            )
            _, self.cat_features = compute_X_cont(self.X_train, self.options["xtypes"])
        # Center and scale X and y
        (
            self.X_norma,
            self.y_norma,
            self.X_offset,
            self.y_mean,
            self.X_scale,
            self.y_std,
        ) = standardization(X, y)

        if not self.options["eval_noise"]:
            self.optimal_noise = np.array(self.options["noise0"])
        elif self.options["use_het_noise"]:
            # hetGP works with unique design variables when noise variance are not given
            (
                self.X_norma,
                index_unique,
                nt_reps,
            ) = np.unique(self.X_norma, return_inverse=True, return_counts=True, axis=0)
            self.nt = self.X_norma.shape[0]

            # computing the mean of the output per unique design variable (see Binois et al., 2018)
            y_norma_unique = []
            for i in range(self.nt):
                y_norma_unique.append(np.mean(self.y_norma[index_unique == i]))
            # pointwise sensible estimates of the noise variances (see Ankenman et al., 2010)
            self.optimal_noise = self.options["noise0"] * np.ones(self.nt)
            for i in range(self.nt):
                diff = self.y_norma[index_unique == i] - y_norma_unique[i]
                if np.sum(diff**2) != 0.0:
                    self.optimal_noise[i] = np.std(diff, ddof=1) ** 2
            self.optimal_noise = self.optimal_noise / nt_reps
            self.y_norma = y_norma_unique
        if self.options["categorical_kernel"] is None:
            # Calculate matrix of distances D between samples
            D, self.ij = cross_distances(self.X_norma)

        if np.min(np.sum(np.abs(D), axis=1)) == 0.0:
            warnings.warn(
                "Warning: multiple x input features have the same value (at least same row twice)."
            )

        ####
        # Regression matrix and parameters
        self.F = self._regression_types[self.options["poly"]](self.X_norma)
        n_samples_F = self.F.shape[0]
        if self.F.ndim > 1:
            p = self.F.shape[1]
        else:
            p = 1
        self._check_F(n_samples_F, p)

        # Optimization
        (
            self.optimal_rlf_value,
            self.optimal_par,
            self.optimal_theta,
        ) = self._optimize_hyperparam(D)
        if self.name in ["MGP"]:
            self._specific_train()
        else:
            if self.options["eval_noise"] and not self.options["use_het_noise"]:
                self.optimal_noise = self.optimal_theta[-1]
                self.optimal_theta = self.optimal_theta[:-1]
        # if self.name != "MGP":
        #     del self.y_norma, self.D

    def _train(self):
        """
        Train the model
        """
        # outputs['sol'] = self.sol

        self._new_train()

    def _matrix_data_corr(
        self,
        corr,
        xtypes,
        theta,
        theta_bounds,
        dx,
        Lij,
        n_levels,
        cat_features,
        cat_kernel,
        x=None,
    ):
        """
        matrix kernel correlation model.

        Parameters
        ----------
        corr: correlation_types
            - The autocorrelation model
        xtypes: np.ndarray [dim]
                -the types (FLOAT,ORD,ENUM) of the input variables
        theta : list[small_d * n_comp]
            Hyperparameters of the correlation model
        dx: np.ndarray[n_obs * (n_obs - 1) / 2, n_comp]
            - The gower_componentwise_distances between the samples.
        Lij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
                - The levels corresponding to the indices i and j of the vectors in X.
        n_levels: np.ndarray
                - The number of levels for every categorical variable.
        cat_features: np.ndarray [dim]
            -  Indices of the categorical input dimensions.
         cat_kernel : string
             - The kernel to use for categorical inputs. Only for non continuous Kriging",
        x : np.ndarray[n_obs , n_comp]
            - The input instead of dx for homo_hs prediction
        Returns
        -------
        r: np.ndarray[n_obs * (n_obs - 1) / 2,1]
            An array containing the values of the autocorrelation model.
        """
        _correlation_types = {
            "abs_exp": abs_exp,
            "squar_exp": squar_exp,
            "act_exp": act_exp,
            "matern52": matern52,
            "matern32": matern32,
        }
        r = np.zeros((dx.shape[0], 1))
        n_components = dx.shape[1]
        nx = self.nx
        nlevels = n_levels
        try:
            cat_kernel_comps = self.options["cat_kernel_comps"]
            if cat_kernel_comps is not None:
                nlevels = np.array(cat_kernel_comps)
        except KeyError:
            cat_kernel_comps = None
        try:
            ncomp = self.options["n_comp"]
            try:
                self.pls_coeff_cont
            except AttributeError:
                self.pls_coeff_cont = []
        except KeyError:
            cat_kernel_comps = None
            ncomp = 1e5
        theta_cont_features = np.zeros((len(theta), 1), dtype=bool)
        theta_cat_features = np.zeros((len(theta), len(nlevels)), dtype=bool)
        i = 0
        j = 0
        n_theta_cont = 0
        for feat in cat_features:
            if feat:
                if cat_kernel in [EXP_HOMO_HSPHERE_KERNEL, HOMO_HSPHERE_KERNEL]:
                    theta_cont_features[
                        j : j + int(nlevels[i] * (nlevels[i] - 1) / 2)
                    ] = False
                    theta_cat_features[
                        j : j + int(nlevels[i] * (nlevels[i] - 1) / 2), i
                    ] = [True] * int(nlevels[i] * (nlevels[i] - 1) / 2)
                    j += int(nlevels[i] * (nlevels[i] - 1) / 2)
                i += 1
            else:
                if cat_kernel in [EXP_HOMO_HSPHERE_KERNEL, HOMO_HSPHERE_KERNEL]:
                    if n_theta_cont < ncomp:
                        theta_cont_features[j] = True
                        j += 1
                        n_theta_cont += 1
        # Sampling points X and y
        X = self.training_points[None][0][0]
        y = self.training_points[None][0][1]

        if cat_kernel == CONT_RELAX_KERNEL:
            from smt.applications.mixed_integer import unfold_with_enum_mask

            X_pls_space = unfold_with_enum_mask(xtypes, X)
            nx = len(theta)

        elif cat_kernel == GOWER_KERNEL:
            X_pls_space = np.copy(X)
        else:
            X_pls_space, _ = compute_X_cont(X, xtypes)
            d_cont = dx[:, np.logical_not(cat_features)]
        if cat_kernel_comps is not None or ncomp < 1e5:
            ###Modifier la condition : if PLS cont
            if self.pls_coeff_cont == []:
                X, y = self._compute_pls(X_pls_space.copy(), y.copy())
                self.pls_coeff_cont = self.coeff_pls
            if cat_kernel == CONT_RELAX_KERNEL or cat_kernel == GOWER_KERNEL:
                d = componentwise_distance_PLS(
                    dx,
                    corr,
                    self.options["n_comp"],
                    self.pls_coeff_cont,
                    theta=None,
                    return_derivative=False,
                )
                r = _correlation_types[corr](theta, d)
                return r
            else:
                d_cont = componentwise_distance_PLS(
                    d_cont,
                    corr,
                    self.options["n_comp"],
                    self.pls_coeff_cont,
                    theta=None,
                    return_derivative=False,
                )
        else:
            d = componentwise_distance(
                dx,
                corr,
                nx,
                theta=None,
                return_derivative=False,
            )
            if cat_kernel != CONT_RELAX_KERNEL:
                d_cont = d[:, np.logical_not(cat_features)]

        if cat_kernel == CONT_RELAX_KERNEL or cat_kernel == GOWER_KERNEL:
            r = _correlation_types[corr](theta, d)
            return r

        theta_cont = theta[theta_cont_features[:, 0]]
        r_cont = _correlation_types[corr](theta_cont, d_cont)
        r_cat = np.copy(r_cont) * 0
        r = np.copy(r_cont)
        ##Theta_cat_i loop
        try:
            self.coeff_pls_cat
        except AttributeError:
            self.coeff_pls_cat = []
        for i in range(len(nlevels)):
            theta_cat = theta[theta_cat_features[:, i]]
            if cat_kernel == EXP_HOMO_HSPHERE_KERNEL:
                theta_cat = theta_cat * (0.5 * np.pi / theta_bounds[1])
            elif cat_kernel == HOMO_HSPHERE_KERNEL:
                theta_cat = theta_cat * (2.0 * np.pi / theta_bounds[1])
            Theta_mat = np.zeros((nlevels[i], nlevels[i]))
            L = np.zeros((nlevels[i], nlevels[i]))
            v = 0
            for j in range(nlevels[i]):
                for k in range(nlevels[i] - j):
                    if j == k + j:
                        Theta_mat[j, k + j] = 1
                    else:
                        Theta_mat[j, k + j] = theta_cat[v]
                        Theta_mat[k + j, j] = theta_cat[v]
                        v = v + 1

            for j in range(nlevels[i]):
                for k in range(nlevels[i] - j):
                    if j == k + j:
                        if j == 0:
                            L[j, k + j] = 1

                        else:
                            L[j, k + j] = 1
                            for l in range(j):
                                L[j, k + j] = L[j, k + j] * np.sin(Theta_mat[j, l])

                    else:
                        if j == 0:
                            L[k + j, j] = np.cos(Theta_mat[k, 0])
                        else:
                            L[k + j, j] = np.cos(Theta_mat[k + j, j])
                            for l in range(j):
                                L[k + j, j] = L[k + j, j] * np.sin(Theta_mat[k + j, l])

            T = np.dot(L, L.T)

            if cat_kernel == EXP_HOMO_HSPHERE_KERNEL:
                T = (T - 1) * theta_bounds[1] / 2
                T = np.exp(2 * T)
            k = (1 + np.exp(-theta_bounds[1])) / np.exp(-theta_bounds[0])
            T = (T + np.exp(-theta_bounds[1])) / (k)

            if cat_kernel_comps is not None:
                # Sampling points X and y
                X = self.training_points[None][0][0]
                y = self.training_points[None][0][1]
                X_icat = X[:, cat_features]
                X_icat = X_icat[:, i]
                old_n_comp = (
                    self.options["n_comp"] if "n_comp" in self.options else None
                )
                self.options["n_comp"] = int(nlevels[i] / 2 * (nlevels[i] - 1))
                X_full_space = compute_X_cross(X_icat, n_levels[i])
                try:
                    self.coeff_pls = self.coeff_pls_cat[i]
                except IndexError:
                    _, _ = self._compute_pls(X_full_space.copy(), y.copy())
                    self.coeff_pls_cat.append(self.coeff_pls)

                if x is not None:
                    x_icat = x[:, cat_features]
                    x_icat = x_icat[:, i]
                    x_full_space = compute_X_cross(x_icat, n_levels[i])
                    dx_cat_i = cross_levels_homo_space(
                        x_full_space, self.ij, y=X_full_space
                    )
                else:
                    dx_cat_i = cross_levels_homo_space(X_full_space, self.ij)

                d_cat_i = componentwise_distance_PLS(
                    dx_cat_i,
                    "squar_exp",
                    self.options["n_comp"],
                    self.coeff_pls,
                    theta=None,
                    return_derivative=False,
                )

            for k in range(np.shape(Lij[i])[0]):
                indi = int(Lij[i][k][0])
                indj = int(Lij[i][k][1])

                if indi == indj:
                    r_cat[k] = 1.0
                else:
                    if cat_kernel in [EXP_HOMO_HSPHERE_KERNEL, HOMO_HSPHERE_KERNEL]:
                        if cat_kernel_comps is not None:
                            Theta_i_red = np.zeros(
                                int((nlevels[i] - 1) * nlevels[i] / 2)
                            )
                            indmatvec = 0
                            for j in range(nlevels[i]):
                                for l in range(nlevels[i]):
                                    if l > j:
                                        Theta_i_red[indmatvec] = T[j, l]
                                        indmatvec += 1
                            kval_cat = 0
                            for indijk in range(len(Theta_i_red)):
                                kval_cat += np.multiply(
                                    Theta_i_red[indijk], d_cat_i[k : k + 1][0][indijk]
                                )
                            r_cat[k] = kval_cat
                        else:
                            r_cat[k] = T[indi, indj]

            r = np.multiply(r, r_cat)
            if cat_kernel_comps is not None:
                if old_n_comp == None:
                    self.options._dict.pop("n_comp", None)
                else:
                    self.options["n_comp"] = old_n_comp
        return r

    def _reduced_likelihood_function(self, theta):
        """
        This function determines the BLUP parameters and evaluates the reduced
        likelihood function for the given autocorrelation parameters theta.
        Maximizing this function wrt the autocorrelation parameters theta is
        equivalent to maximizing the likelihood of the assumed joint Gaussian
        distribution of the observations y evaluated onto the design of
        experiments X.

        Parameters
        ----------
        theta: list(n_comp), optional
            - An array containing the autocorrelation parameters at which the
              Gaussian Process model parameters should be determined.

        Returns
        -------
        reduced_likelihood_function_value: real
            - The value of the reduced likelihood function associated to the
              given autocorrelation parameters theta.
        par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters:
            sigma2
            Gaussian Process variance.
            beta
            Generalized least-squares regression weights for
            Universal Kriging or for Ordinary Kriging.
            gamma
            Gaussian Process weights.
            C
            Cholesky decomposition of the correlation matrix [R].
            Ft
            Solution of the linear equation system : [R] x Ft = F
            Q, G
            QR decomposition of the matrix Ft.
        """
        # Initialize output

        reduced_likelihood_function_value = -np.inf
        par = {}
        # Set up R
        nugget = self.options["nugget"]
        if self.options["eval_noise"]:
            nugget = 0

        noise = self.noise0
        tmp_var = theta
        if self.options["use_het_noise"]:
            noise = self.optimal_noise
        if self.options["eval_noise"] and not self.options["use_het_noise"]:
            theta = tmp_var[0 : self.D.shape[1]]
            noise = tmp_var[self.D.shape[1] :]
        if self.options["categorical_kernel"] is not None:
            dx = self.D
            if self.options["categorical_kernel"] == CONT_RELAX_KERNEL:
                from smt.applications.mixed_integer import unfold_with_enum_mask

                X2 = unfold_with_enum_mask(
                    self.options["xtypes"], self.training_points[None][0][0]
                )
                (
                    self.X2_norma,
                    _,
                    self.X2_offset,
                    _,
                    self.X2_scale,
                    _,
                ) = standardization(X2, self.training_points[None][0][1])
                dx, _ = cross_distances(self.X2_norma)

            r = self._matrix_data_corr(
                corr=self.options["corr"],
                xtypes=self.options["xtypes"],
                theta=theta,
                theta_bounds=self.options["theta_bounds"],
                dx=dx,
                Lij=self.Lij,
                n_levels=self.n_levels,
                cat_features=self.cat_features,
                cat_kernel=self.options["categorical_kernel"],
            ).reshape(-1, 1)
        else:
            r = self._correlation_types[self.options["corr"]](theta, self.D).reshape(
                -1, 1
            )

        R = np.eye(self.nt) * (1.0 + nugget + noise)
        R[self.ij[:, 0], self.ij[:, 1]] = r[:, 0]
        R[self.ij[:, 1], self.ij[:, 0]] = r[:, 0]
        # Cholesky decomposition of R

        try:
            C = linalg.cholesky(R, lower=True)
        except (linalg.LinAlgError, ValueError) as e:
            print("exception : ", e)
            print(np.linalg.eig(R)[0])
            return reduced_likelihood_function_value, par

        # Get generalized least squared solution
        Ft = linalg.solve_triangular(C, self.F, lower=True)
        Q, G = linalg.qr(Ft, mode="economic")
        sv = linalg.svd(G, compute_uv=False)
        rcondG = sv[-1] / sv[0]
        if rcondG < 1e-10:
            # Check F
            sv = linalg.svd(self.F, compute_uv=False)
            condF = sv[0] / sv[-1]
            if condF > 1e15:
                raise Exception(
                    "F is too ill conditioned. Poor combination "
                    "of regression model and observations."
                )

            else:
                # Ft is too ill conditioned, get out (try different theta)
                return reduced_likelihood_function_value, par

        Yt = linalg.solve_triangular(C, self.y_norma, lower=True)
        beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
        rho = Yt - np.dot(Ft, beta)

        # The determinant of R is equal to the squared product of the diagonal
        # elements of its Cholesky decomposition C
        detR = (np.diag(C) ** (2.0 / self.nt)).prod()
        # Compute/Organize output
        p = 0
        q = 0
        if self.name in ["MFK", "MFKPLS", "MFKPLSK"]:
            p = self.p
            q = self.q
        sigma2 = (rho**2.0).sum(axis=0) / (self.nt - p - q)
        reduced_likelihood_function_value = -(self.nt - p - q) * np.log10(
            sigma2.sum()
        ) - self.nt * np.log10(detR)
        par["sigma2"] = sigma2 * self.y_std**2.0
        par["beta"] = beta
        par["gamma"] = linalg.solve_triangular(C.T, rho)
        par["C"] = C
        par["Ft"] = Ft
        par["G"] = G
        par["Q"] = Q

        if self.name in ["MGP"]:
            reduced_likelihood_function_value += self._reduced_log_prior(theta)

        # A particular case when f_min_cobyla fail
        if (self.best_iteration_fail is not None) and (
            not np.isinf(reduced_likelihood_function_value)
        ):

            if reduced_likelihood_function_value > self.best_iteration_fail:
                self.best_iteration_fail = reduced_likelihood_function_value
                self._thetaMemory = np.array(tmp_var)

        elif (self.best_iteration_fail is None) and (
            not np.isinf(reduced_likelihood_function_value)
        ):
            self.best_iteration_fail = reduced_likelihood_function_value
            self._thetaMemory = np.array(tmp_var)
        if reduced_likelihood_function_value > 1e15:
            reduced_likelihood_function_value = 1e15
        return reduced_likelihood_function_value, par

    def _reduced_likelihood_gradient(self, theta):
        """
        Evaluates the reduced_likelihood_gradient at a set of hyperparameters.

        Parameters
        ---------
        theta : list(n_comp), optional
            - An array containing the autocorrelation parameters at which the
              Gaussian Process model parameters should be determined.

        Returns
        -------
        grad_red : np.ndarray (dim,1)
            Derivative of the reduced_likelihood
        par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters:
            sigma2
            Gaussian Process variance.
            beta
            Generalized least-squares regression weights for
            Universal Kriging or for Ordinary Kriging.
            gamma
            Gaussian Process weights.
            C
            Cholesky decomposition of the correlation matrix [R].
            Ft
            Solution of the linear equation system : [R] x Ft = F
            Q, G
            QR decomposition of the matrix Ft.
            dr
            List of all the correlation matrix derivative
            tr
            List of all the trace part in the reduce likelihood derivatives
            dmu
            List of all the mean derivatives
            arg
            List of all minus_Cinv_dRdomega_gamma
            dsigma
            List of all sigma derivatives
        """
        red, par = self._reduced_likelihood_function(theta)

        C = par["C"]
        gamma = par["gamma"]
        Q = par["Q"]
        G = par["G"]
        sigma_2 = par["sigma2"]

        nb_theta = len(theta)
        grad_red = np.zeros(nb_theta)

        dr_all = []
        tr_all = []
        dmu_all = []
        arg_all = []
        dsigma_all = []
        dbeta_all = []
        for i_der in range(nb_theta):
            # Compute R derivatives
            dr = self._correlation_types[self.options["corr"]](
                theta, self.D, grad_ind=i_der
            )

            dr_all.append(dr)

            dR = np.zeros((self.nt, self.nt))
            dR[self.ij[:, 0], self.ij[:, 1]] = dr[:, 0]
            dR[self.ij[:, 1], self.ij[:, 0]] = dr[:, 0]

            # Compute beta derivatives
            Cinv_dR_gamma = linalg.solve_triangular(C, np.dot(dR, gamma), lower=True)
            dbeta = -linalg.solve_triangular(G, np.dot(Q.T, Cinv_dR_gamma))
            arg_all.append(Cinv_dR_gamma)

            dbeta_all.append(dbeta)

            # Compute mu derivatives
            dmu = np.dot(self.F, dbeta)
            dmu_all.append(dmu)

            # Compute log(detR) derivatives
            tr_1 = linalg.solve_triangular(C, dR, lower=True)
            tr = linalg.solve_triangular(C.T, tr_1)
            tr_all.append(tr)

            # Compute Sigma2 Derivatives
            dsigma_2 = (
                (1 / self.nt)
                * (
                    -dmu.T.dot(gamma)
                    - gamma.T.dot(dmu)
                    - np.dot(gamma.T, dR.dot(gamma))
                )
                * self.y_std**2.0
            )
            dsigma_all.append(dsigma_2)

            # Compute reduced log likelihood derivatives
            grad_red[i_der] = (
                -self.nt / np.log(10) * (dsigma_2 / sigma_2 + np.trace(tr) / self.nt)
            )

        par["dr"] = dr_all
        par["tr"] = tr_all
        par["dmu"] = dmu_all
        par["arg"] = arg_all
        par["dsigma"] = dsigma_all
        par["dbeta_all"] = dbeta_all

        grad_red = np.atleast_2d(grad_red).T

        if self.name in ["MGP"]:
            grad_red += self._reduced_log_prior(theta, grad=True)
        return grad_red, par

    def _reduced_likelihood_hessian(self, theta):
        """
        Evaluates the reduced_likelihood_gradient at a set of hyperparameters.

        Parameters
        ----------
        theta : list(n_comp), optional
            - An array containing the autocorrelation parameters at which the
              Gaussian Process model parameters should be determined.

        Returns
        -------
        hess : np.ndarray
            Hessian values.
        hess_ij: np.ndarray [nb_theta * (nb_theta + 1) / 2, 2]
            - The indices i and j of the vectors in theta associated to the hessian in hess.
        par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters:
            sigma2
            Gaussian Process variance.
            beta
            Generalized least-squared regression weights for
            Universal Kriging or for Ordinary Kriging.
            gamma
            Gaussian Process weights.
            C
            Cholesky decomposition of the correlation matrix [R].
            Ft
            Solution of the linear equation system : [R] x Ft = F
            Q, G
            QR decomposition of the matrix Ft.
            dr
            List of all the correlation matrix derivative
            tr
            List of all the trace part in the reduce likelihood derivatives
            dmu
            List of all the mean derivatives
            arg
            List of all minus_Cinv_dRdomega_gamma
            dsigma
            List of all sigma derivatives
        """
        dred, par = self._reduced_likelihood_gradient(theta)

        C = par["C"]
        gamma = par["gamma"]
        Q = par["Q"]
        G = par["G"]
        sigma_2 = par["sigma2"]

        nb_theta = len(theta)

        dr_all = par["dr"]
        tr_all = par["tr"]
        dmu_all = par["dmu"]
        arg_all = par["arg"]
        dsigma = par["dsigma"]
        Rinv_dRdomega_gamma_all = []
        Rinv_dmudomega_all = []

        n_val_hess = nb_theta * (nb_theta + 1) // 2
        hess_ij = np.zeros((n_val_hess, 2), dtype=np.int32)
        hess = np.zeros((n_val_hess, 1))
        ind_1 = 0
        if self.name in ["MGP"]:
            log_prior = self._reduced_log_prior(theta, hessian=True)

        for omega in range(nb_theta):
            ind_0 = ind_1
            ind_1 = ind_0 + nb_theta - omega
            hess_ij[ind_0:ind_1, 0] = omega
            hess_ij[ind_0:ind_1, 1] = np.arange(omega, nb_theta)

            dRdomega = np.zeros((self.nt, self.nt))
            dRdomega[self.ij[:, 0], self.ij[:, 1]] = dr_all[omega][:, 0]
            dRdomega[self.ij[:, 1], self.ij[:, 0]] = dr_all[omega][:, 0]

            dmudomega = dmu_all[omega]
            Cinv_dmudomega = linalg.solve_triangular(C, dmudomega, lower=True)
            Rinv_dmudomega = linalg.solve_triangular(C.T, Cinv_dmudomega)
            Rinv_dmudomega_all.append(Rinv_dmudomega)
            Rinv_dRdomega_gamma = linalg.solve_triangular(C.T, arg_all[omega])
            Rinv_dRdomega_gamma_all.append(Rinv_dRdomega_gamma)

            for i, eta in enumerate(hess_ij[ind_0:ind_1, 1]):
                dRdeta = np.zeros((self.nt, self.nt))
                dRdeta[self.ij[:, 0], self.ij[:, 1]] = dr_all[eta][:, 0]
                dRdeta[self.ij[:, 1], self.ij[:, 0]] = dr_all[eta][:, 0]

                dr_eta_omega = self._correlation_types[self.options["corr"]](
                    theta, self.D, grad_ind=omega, hess_ind=eta
                )
                dRdetadomega = np.zeros((self.nt, self.nt))
                dRdetadomega[self.ij[:, 0], self.ij[:, 1]] = dr_eta_omega[:, 0]
                dRdetadomega[self.ij[:, 1], self.ij[:, 0]] = dr_eta_omega[:, 0]

                # Compute beta second derivatives
                dRdeta_Rinv_dmudomega = np.dot(dRdeta, Rinv_dmudomega)

                dmudeta = dmu_all[eta]
                Cinv_dmudeta = linalg.solve_triangular(C, dmudeta, lower=True)
                Rinv_dmudeta = linalg.solve_triangular(C.T, Cinv_dmudeta)
                dRdomega_Rinv_dmudeta = np.dot(dRdomega, Rinv_dmudeta)

                dRdeta_Rinv_dRdomega_gamma = np.dot(dRdeta, Rinv_dRdomega_gamma)

                Rinv_dRdeta_gamma = linalg.solve_triangular(C.T, arg_all[eta])
                dRdomega_Rinv_dRdeta_gamma = np.dot(dRdomega, Rinv_dRdeta_gamma)

                dRdetadomega_gamma = np.dot(dRdetadomega, gamma)

                beta_sum = (
                    dRdeta_Rinv_dmudomega
                    + dRdomega_Rinv_dmudeta
                    + dRdeta_Rinv_dRdomega_gamma
                    + dRdomega_Rinv_dRdeta_gamma
                    - dRdetadomega_gamma
                )

                Qt_Cinv_beta_sum = np.dot(
                    Q.T, linalg.solve_triangular(C, beta_sum, lower=True)
                )
                dbetadetadomega = linalg.solve_triangular(G, Qt_Cinv_beta_sum)

                # Compute mu second derivatives
                dmudetadomega = np.dot(self.F, dbetadetadomega)

                # Compute sigma2 second derivatives
                sigma_arg_1 = (
                    -np.dot(dmudetadomega.T, gamma)
                    + np.dot(dmudomega.T, Rinv_dRdeta_gamma)
                    + np.dot(dmudeta.T, Rinv_dRdomega_gamma)
                )

                sigma_arg_2 = (
                    -np.dot(gamma.T, dmudetadomega)
                    + np.dot(gamma.T, dRdeta_Rinv_dmudomega)
                    + np.dot(gamma.T, dRdomega_Rinv_dmudeta)
                )

                sigma_arg_3 = np.dot(dmudeta.T, Rinv_dmudomega) + np.dot(
                    dmudomega.T, Rinv_dmudeta
                )

                sigma_arg_4_in = (
                    -dRdetadomega_gamma
                    + dRdeta_Rinv_dRdomega_gamma
                    + dRdomega_Rinv_dRdeta_gamma
                )
                sigma_arg_4 = np.dot(gamma.T, sigma_arg_4_in)

                dsigma2detadomega = (
                    (1 / self.nt)
                    * (sigma_arg_1 + sigma_arg_2 + sigma_arg_3 + sigma_arg_4)
                    * self.y_std**2.0
                )

                # Compute Hessian
                dreddetadomega_tr_1 = np.trace(np.dot(tr_all[eta], tr_all[omega]))

                dreddetadomega_tr_2 = np.trace(
                    linalg.solve_triangular(
                        C.T, linalg.solve_triangular(C, dRdetadomega, lower=True)
                    )
                )

                dreddetadomega_arg1 = (self.nt / sigma_2) * (
                    dsigma2detadomega - (1 / sigma_2) * dsigma[omega] * dsigma[eta]
                )
                dreddetadomega = (
                    -(dreddetadomega_arg1 - dreddetadomega_tr_1 + dreddetadomega_tr_2)
                    / self.nt
                )

                hess[ind_0 + i, 0] = self.nt / np.log(10) * dreddetadomega

                if self.name in ["MGP"] and eta == omega:
                    hess[ind_0 + i, 0] += log_prior[eta]
            par["Rinv_dR_gamma"] = Rinv_dRdomega_gamma_all
            par["Rinv_dmu"] = Rinv_dmudomega_all
        return hess, hess_ij, par

    def _predict_values(self, x):
        """
        Evaluates the model at a set of points.

        Parameters
        ----------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        y : np.ndarray
            Evaluation point output variable values
        """
        # Initialization
        n_eval, n_features_x = x.shape
        if self.options["categorical_kernel"] is not None:
            dx = gower_componentwise_distances(
                x, y=np.copy(self.X_train), xtypes=self.options["xtypes"]
            )

            d = componentwise_distance(
                dx,
                self.options["corr"],
                self.nx,
                theta=None,
                return_derivative=False,
            )
            if self.options["categorical_kernel"] is not None:
                _, ij = cross_distances(x, self.X_train)
                Lij, _ = cross_levels(
                    X=x, ij=ij, xtypes=self.options["xtypes"], y=self.X_train
                )
                self.ij = ij
                if self.options["categorical_kernel"] == CONT_RELAX_KERNEL:
                    Xpred = unfold_with_enum_mask(self.options["xtypes"], x)
                    Xpred_norma = (Xpred - self.X2_offset) / self.X2_scale
                    # Get pairwise componentwise L1-distances to the input training set
                    dx = differences(Xpred_norma, Y=self.X2_norma.copy())
                r = self._matrix_data_corr(
                    corr=self.options["corr"],
                    xtypes=self.options["xtypes"],
                    theta=self.optimal_theta,
                    theta_bounds=self.options["theta_bounds"],
                    dx=dx,
                    Lij=Lij,
                    n_levels=self.n_levels,
                    cat_features=self.cat_features,
                    cat_kernel=self.options["categorical_kernel"],
                    x=x,
                ).reshape(n_eval, self.nt)

            X_cont, _ = compute_X_cont(x, self.options["xtypes"])
            X_cont = (X_cont - self.X_offset) / self.X_scale

        else:
            X_cont = (x - self.X_offset) / self.X_scale
            # Get pairwise componentwise L1-distances to the input training set
            dx = differences(X_cont, Y=self.X_norma.copy())
            d = self._componentwise_distance(dx)
            # Compute the correlation function
            r = self._correlation_types[self.options["corr"]](
                self.optimal_theta, d
            ).reshape(n_eval, self.nt)
            y = np.zeros(n_eval)
        # Compute the regression function
        f = self._regression_types[self.options["poly"]](X_cont)
        # Scaled predictor
        y_ = np.dot(f, self.optimal_par["beta"]) + np.dot(r, self.optimal_par["gamma"])
        # Predictor
        y = (self.y_mean + self.y_std * y_).ravel()
        return y

    def _predict_derivatives(self, x, kx):
        """
        Evaluates the derivatives at a set of points.

        Parameters
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        kx : int
            The 0-based index of the input variable with respect to which derivatives are desired.

        Returns
        -------
        y : np.ndarray
            Derivative values.
        """
        # Initialization
        n_eval, n_features_x = x.shape

        x = (x - self.X_offset) / self.X_scale
        # Get pairwise componentwise L1-distances to the input training set

        dx = differences(x, Y=self.X_norma.copy())
        d = self._componentwise_distance(dx)
        dd = self._componentwise_distance(
            dx, theta=self.optimal_theta, return_derivative=True
        )

        # Compute the correlation function
        derivative_dic = {"dx": dx, "dd": dd}

        r, dr = self._correlation_types[self.options["corr"]](
            self.optimal_theta, d, derivative_params=derivative_dic
        )
        r = r.reshape(n_eval, self.nt)
        dr = (dr.T).reshape(n_features_x, self.nt, n_eval)

        if self.options["poly"] == "constant":
            df = np.zeros((1, self.nx))
        elif self.options["poly"] == "linear":
            df = np.zeros((self.nx + 1, self.nx))
            df[1:, :] = np.eye(self.nx)
        else:
            raise ValueError(
                "The derivative is only available for ordinary kriging or "
                + "universal kriging using a linear trend"
            )

        # Beta and gamma = R^-1(y-FBeta)
        beta = self.optimal_par["beta"]
        gamma = self.optimal_par["gamma"]
        df_dx = np.dot(df.T, beta)
        d_dx = x[:, kx].reshape((n_eval, 1)) - self.X_norma[:, kx].reshape((1, self.nt))
        if self.name != "Kriging" and "KPLSK" not in self.name:
            theta = np.sum(self.optimal_theta * self.coeff_pls**2, axis=1)
        else:
            theta = self.optimal_theta
        y = (df_dx[kx] + np.dot(dr[kx].T, gamma)) * self.y_std / self.X_scale[kx]
        return y

    def _predict_variances(self, x):
        """
        Provide uncertainty of the model at a set of points
        Parameters
        ----------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        Returns
        -------
        MSE : np.ndarray
            Evaluation point output variable MSE
        """
        # Initialization
        n_eval, n_features_x = x.shape
        X_cont = x
        if self.options["categorical_kernel"] is not None:

            dx = gower_componentwise_distances(
                x, y=np.copy(self.X_train), xtypes=self.options["xtypes"]
            )
            d = componentwise_distance(
                dx,
                self.options["corr"],
                self.nx,
                theta=None,
                return_derivative=False,
            )
            if self.options["categorical_kernel"] is not None:

                _, ij = cross_distances(x, self.X_train)
                Lij, _ = cross_levels(
                    X=x, ij=ij, xtypes=self.options["xtypes"], y=self.X_train
                )
                self.ij = ij
                if self.options["categorical_kernel"] == CONT_RELAX_KERNEL:
                    from smt.applications.mixed_integer import unfold_with_enum_mask

                    Xpred = unfold_with_enum_mask(self.options["xtypes"], x)
                    Xpred_norma = (Xpred - self.X2_offset) / self.X2_scale

                    # Get pairwise componentwise L1-distances to the input training set
                    dx = differences(Xpred_norma, Y=self.X2_norma.copy())
                r = self._matrix_data_corr(
                    corr=self.options["corr"],
                    xtypes=self.options["xtypes"],
                    theta=self.optimal_theta,
                    theta_bounds=self.options["theta_bounds"],
                    dx=dx,
                    Lij=Lij,
                    n_levels=self.n_levels,
                    cat_features=self.cat_features,
                    cat_kernel=self.options["categorical_kernel"],
                    x=x,
                ).reshape(n_eval, self.nt)
            X_cont, _ = compute_X_cont(x, self.options["xtypes"])
            X_cont = (X_cont - self.X_offset) / self.X_scale
        else:
            x = (x - self.X_offset) / self.X_scale
            X_cont = np.copy(x)
            # Get pairwise componentwise L1-distances to the input training set
            dx = differences(x, Y=self.X_norma.copy())
            d = self._componentwise_distance(dx)
            # Compute the correlation function
            r = self._correlation_types[self.options["corr"]](
                self.optimal_theta, d
            ).reshape(n_eval, self.nt)

        C = self.optimal_par["C"]
        rt = linalg.solve_triangular(C, r.T, lower=True)

        u = linalg.solve_triangular(
            self.optimal_par["G"].T,
            np.dot(self.optimal_par["Ft"].T, rt)
            - self._regression_types[self.options["poly"]](X_cont).T,
        )
        A = self.optimal_par["sigma2"]
        B = 1.0 - (rt**2.0).sum(axis=0) + (u**2.0).sum(axis=0)
        MSE = np.einsum("i,j -> ji", A, B)
        # Mean Squared Error might be slightly negative depending on
        # machine precision: force to zero!
        MSE[MSE < 0.0] = 0.0
        return MSE

    def _predict_variance_derivatives(self, x):
        """
        Provide the derivative of the variance of the model at a set of points
        Parameters
        -----------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        Returns
        -------
         derived_variance:  np.ndarray
             The jacobian of the variance of the kriging model
        """

        # Initialization
        n_eval, n_features_x = x.shape
        x = (x - self.X_offset) / self.X_scale
        theta = self.optimal_theta
        # Get pairwise componentwise L1-distances to the input training set
        dx = differences(x, Y=self.X_norma.copy())
        d = self._componentwise_distance(dx)
        dd = self._componentwise_distance(
            dx, theta=self.optimal_theta, return_derivative=True
        )
        derivative_dic = {"dx": dx, "dd": dd}

        sigma2 = self.optimal_par["sigma2"]
        C = self.optimal_par["C"]

        # p1 : derivative of (rt**2.0).sum(axis=0)
        r, dr = self._correlation_types[self.options["corr"]](
            theta, d, derivative_params=derivative_dic
        )
        rt = linalg.solve_triangular(C, r, lower=True)
        invKr = linalg.solve_triangular(C.T, rt)
        p1 = 2 * np.dot(dr.T, invKr).T

        # p2 : derivative of (u**2.0).sum(axis=0)
        f_x = self._regression_types[self.options["poly"]](x).T
        F = self.F
        rho2 = linalg.solve_triangular(C, F, lower=True)
        invKF = linalg.solve_triangular(C.T, rho2)

        A = f_x.T - np.dot(r.T, invKF)
        B = np.dot(F.T, invKF)
        rho3 = linalg.cholesky(B, lower=True)
        invBAt = linalg.solve_triangular(rho3, A.T, lower=True)
        D = linalg.solve_triangular(rho3.T, invBAt)

        if self.options["poly"] == "constant":
            df = np.zeros((1, self.nx))
        elif self.options["poly"] == "linear":
            df = np.zeros((self.nx + 1, self.nx))
            df[1:, :] = np.eye(self.nx)
        else:
            raise ValueError(
                "The derivative is only available for ordinary kriging or "
                + "universal kriging using a linear trend"
            )

        dA = df.T - np.dot(dr.T, invKF)
        p3 = 2 * np.dot(dA, D).T

        # prime : derivative of MSE
        # MSE ~1.0 - (rt**2.0).sum(axis=0) + (u**2.0).sum(axis=0)
        prime = 0 - p1 + p3
        ## scaling factors
        derived_variance = []
        x_std = np.resize(self.X_scale, self.nx)

        for i in range(len(x_std)):
            derived_variance.append(sigma2 * prime.T[i] / x_std[i])

        return np.array(derived_variance).T

    def _optimize_hyperparam(self, D):
        """
        This function evaluates the Gaussian Process model at x.

        Parameters
        ----------
        D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The componentwise cross-spatial-correlation-distance between the
              vectors in X.

        Returns
        -------
        best_optimal_rlf_value: real
            - The value of the reduced likelihood function associated to the
              best autocorrelation parameters theta.
        best_optimal_par: dict()
            - A dictionary containing the requested Gaussian Process model
              parameters.
        best_optimal_theta: list(n_comp) or list(dim)
            - The best hyperparameters found by the optimization.
        """
        # reinitialize optimization best values
        self.best_iteration_fail = None
        self._thetaMemory = None
        # Initialize the hyperparameter-optimization
        if self.name in ["MGP"]:

            def minus_reduced_likelihood_function(theta):
                res = -self._reduced_likelihood_function(theta)[0]
                return res

            def grad_minus_reduced_likelihood_function(theta):
                grad = -self._reduced_likelihood_gradient(theta)[0]
                return grad

        else:

            def minus_reduced_likelihood_function(log10t):
                return -self._reduced_likelihood_function(theta=10.0**log10t)[0]

            def grad_minus_reduced_likelihood_function(log10t):
                log10t_2d = np.atleast_2d(log10t).T
                res = (
                    -np.log(10.0)
                    * (10.0**log10t_2d)
                    * (self._reduced_likelihood_gradient(10.0**log10t_2d)[0])
                )
                return res

        limit, _rhobeg = 15 * len(self.options["theta0"]), 0.5
        exit_function = False
        if "KPLSK" in self.name:
            n_iter = 1
        else:
            n_iter = 0

        for ii in range(n_iter, -1, -1):
            (
                best_optimal_theta,
                best_optimal_rlf_value,
                best_optimal_par,
                constraints,
            ) = (
                [],
                [],
                [],
                [],
            )

            bounds_hyp = []

            self.theta0 = deepcopy(self.options["theta0"])
            for i in range(len(self.theta0)):
                # In practice, in 1D and for X in [0,1], theta^{-2} in [1e-2,infty),
                # i.e. theta in (0,1e1], is a good choice to avoid overfitting.
                # By standardising X in R, X_norm = (X-X_mean)/X_std, then
                # X_norm in [-1,1] if considering one std intervals. This leads
                # to theta in (0,2e1]
                theta_bounds = self.options["theta_bounds"]
                if self.theta0[i] < theta_bounds[0] or self.theta0[i] > theta_bounds[1]:
                    self.theta0[i] = np.random.rand()
                    self.theta0[i] = (
                        self.theta0[i] * (theta_bounds[1] - theta_bounds[0])
                        + theta_bounds[0]
                    )
                    warnings.warn(
                        "Warning: theta0 is out the feasible bounds. A random initialisation is used instead."
                    )

                if self.name in ["MGP"]:  # to be discussed with R. Priem
                    constraints.append(lambda theta, i=i: theta[i] + theta_bounds[1])
                    constraints.append(lambda theta, i=i: theta_bounds[1] - theta[i])
                    bounds_hyp.append((-theta_bounds[1], theta_bounds[1]))
                else:
                    log10t_bounds = np.log10(theta_bounds)
                    constraints.append(lambda log10t, i=i: log10t[i] - log10t_bounds[0])
                    constraints.append(lambda log10t, i=i: log10t_bounds[1] - log10t[i])
                    bounds_hyp.append(log10t_bounds)

            if self.name in ["MGP"]:
                theta0_rand = m_norm.rvs(
                    self.options["prior"]["mean"] * len(self.theta0),
                    self.options["prior"]["var"],
                    1,
                )
                theta0 = self.theta0
            else:
                theta0_rand = np.random.rand(len(self.theta0))
                theta0_rand = (
                    theta0_rand * (log10t_bounds[1] - log10t_bounds[0])
                    + log10t_bounds[0]
                )
                theta0 = np.log10(self.theta0)

            if self.options["categorical_kernel"]:
                self.D = D
            else:
                ##from abs distance to kernel distance
                self.D = self._componentwise_distance(D, opt=ii)

            # Initialization
            k, incr, stop, best_optimal_rlf_value, max_retry = 0, 0, 1, -1e20, 10
            while k < stop:
                # Use specified starting point as first guess
                self.noise0 = np.array(self.options["noise0"])
                noise_bounds = self.options["noise_bounds"]
                if self.options["eval_noise"] and not self.options["use_het_noise"]:
                    self.noise0[self.noise0 == 0.0] = noise_bounds[0]
                    for i in range(len(self.noise0)):
                        if (
                            self.noise0[i] < noise_bounds[0]
                            or self.noise0[i] > noise_bounds[1]
                        ):
                            self.noise0[i] = noise_bounds[0]
                            warnings.warn(
                                "Warning: noise0 is out the feasible bounds. The lowest possible value is used instead."
                            )

                    theta0 = np.concatenate(
                        [theta0, np.log10(np.array([self.noise0]).flatten())]
                    )
                    theta0_rand = np.concatenate(
                        [
                            theta0_rand,
                            np.log10(np.array([self.noise0]).flatten()),
                        ]
                    )

                    for i in range(len(self.noise0)):
                        noise_bounds = np.log10(noise_bounds)
                        constraints.append(
                            lambda log10t: log10t[i + len(self.theta0)]
                            - noise_bounds[0]
                        )
                        constraints.append(
                            lambda log10t: noise_bounds[1]
                            - log10t[i + len(self.theta0)]
                        )
                        bounds_hyp.append(noise_bounds)
                theta_limits = np.repeat(
                    np.log10([theta_bounds]), repeats=len(theta0), axis=0
                )
                theta_all_loops = np.vstack((theta0, theta0_rand))

                if self.options["n_start"] > 1:
                    sampling = LHS(
                        xlimits=theta_limits, criterion="maximin", random_state=41
                    )
                    theta_lhs_loops = sampling(self.options["n_start"])
                    theta_all_loops = np.vstack((theta_all_loops, theta_lhs_loops))

                optimal_theta_res = {"fun": float("inf")}
                try:
                    if self.options["hyper_opt"] == "Cobyla":
                        for theta0_loop in theta_all_loops:
                            optimal_theta_res_loop = optimize.minimize(
                                minus_reduced_likelihood_function,
                                theta0_loop,
                                constraints=[
                                    {"fun": con, "type": "ineq"} for con in constraints
                                ],
                                method="COBYLA",
                                options={
                                    "rhobeg": _rhobeg,
                                    "tol": 1e-4,
                                    "maxiter": limit,
                                },
                            )
                            if optimal_theta_res_loop["fun"] < optimal_theta_res["fun"]:
                                optimal_theta_res = optimal_theta_res_loop

                    elif self.options["hyper_opt"] == "TNC":
                        theta_all_loops = 10**theta_all_loops
                        for theta0_loop in theta_all_loops:
                            optimal_theta_res_loop = optimize.minimize(
                                minus_reduced_likelihood_function,
                                theta0_loop,
                                method="TNC",
                                jac=grad_minus_reduced_likelihood_function,
                                bounds=bounds_hyp,
                                options={"maxiter": 100},
                            )
                            if optimal_theta_res_loop["fun"] < optimal_theta_res["fun"]:
                                optimal_theta_res = optimal_theta_res_loop

                    optimal_theta = optimal_theta_res["x"]

                    if self.name not in ["MGP"]:
                        optimal_theta = 10**optimal_theta

                    optimal_rlf_value, optimal_par = self._reduced_likelihood_function(
                        theta=optimal_theta
                    )
                    # Compare the new optimizer to the best previous one
                    if k > 0:
                        if np.isinf(optimal_rlf_value):
                            stop += 1
                            if incr != 0:
                                return
                            if stop > max_retry:
                                raise ValueError(
                                    "%d attempts to train the model failed" % max_retry
                                )
                        else:
                            if optimal_rlf_value >= self.best_iteration_fail:
                                if optimal_rlf_value > best_optimal_rlf_value:
                                    best_optimal_rlf_value = optimal_rlf_value
                                    best_optimal_par = optimal_par
                                    best_optimal_theta = optimal_theta
                                else:
                                    if (
                                        self.best_iteration_fail
                                        > best_optimal_rlf_value
                                    ):
                                        best_optimal_theta = self._thetaMemory
                                        (
                                            best_optimal_rlf_value,
                                            best_optimal_par,
                                        ) = self._reduced_likelihood_function(
                                            theta=best_optimal_theta
                                        )
                    else:
                        if np.isinf(optimal_rlf_value):
                            stop += 1
                        else:
                            best_optimal_rlf_value = optimal_rlf_value
                            best_optimal_par = optimal_par
                            best_optimal_theta = optimal_theta
                    k += 1
                except ValueError as ve:
                    # raise ve
                    # If iteration is max when fmin_cobyla fail is not reached
                    if self.nb_ill_matrix > 0:
                        self.nb_ill_matrix -= 1
                        k += 1
                        stop += 1
                        # One evaluation objectif function is done at least
                        if self.best_iteration_fail is not None:
                            if self.best_iteration_fail > best_optimal_rlf_value:
                                best_optimal_theta = self._thetaMemory
                                (
                                    best_optimal_rlf_value,
                                    best_optimal_par,
                                ) = self._reduced_likelihood_function(
                                    theta=best_optimal_theta
                                )
                    # Optimization fail
                    elif best_optimal_par == []:
                        print("Optimization failed. Try increasing the ``nugget``")
                        raise ve
                    # Break the while loop
                    else:
                        k = stop + 1
                        print("fmin_cobyla failed but the best value is retained")

            if "KPLSK" in self.name:
                if self.options["eval_noise"]:
                    # best_optimal_theta contains [theta, noise] if eval_noise = True
                    theta = best_optimal_theta[:-1]
                else:
                    # best_optimal_theta contains [theta] if eval_noise = False
                    theta = best_optimal_theta

                if exit_function:
                    return best_optimal_rlf_value, best_optimal_par, best_optimal_theta

                if self.options["corr"] == "squar_exp":
                    self.options["theta0"] = (theta * self.coeff_pls**2).sum(1)
                else:
                    self.options["theta0"] = (theta * np.abs(self.coeff_pls)).sum(1)

                self.options["n_comp"] = int(self.nx)
                limit = 10 * self.options["n_comp"]
                self.best_iteration_fail = None
                exit_function = True
        return best_optimal_rlf_value, best_optimal_par, best_optimal_theta

    def _check_param(self):
        """
        This function checks some parameters of the model
        and amend theta0 if possible (see _amend_theta0_option).
        """
        d = self.options["n_comp"] if "n_comp" in self.options else self.nx

        if self.name in ["KPLS"]:
            if self.options["corr"] not in ["squar_exp", "abs_exp"]:
                raise ValueError(
                    "KPLS only works with a squared exponential or an absolute exponential kernel"
                )
            if (
                self.options["categorical_kernel"]
                not in [EXP_HOMO_HSPHERE_KERNEL, HOMO_HSPHERE_KERNEL]
                and self.name == "KPLS"
            ):
                if self.options["cat_kernel_comps"] is not None:
                    raise ValueError(
                        "cat_kernel_comps option is for homoscedastic kernel."
                    )

        mat_dim = (
            self.options["cat_kernel_comps"]
            if "cat_kernel_comps" in self.options
            else None
        )
        if self.options["categorical_kernel"] in [
            EXP_HOMO_HSPHERE_KERNEL,
            HOMO_HSPHERE_KERNEL,
            CONT_RELAX_KERNEL,
        ]:
            n_comp = self.options["n_comp"] if "n_comp" in self.options else None
            n_param = compute_n_param(
                self.options["xtypes"],
                self.options["categorical_kernel"],
                self.nx,
                d,
                n_comp,
                mat_dim,
            )

            self.options["theta0"] *= np.ones(n_param)

        if len(self.options["theta0"]) != d and self.options[
            "categorical_kernel"
        ] not in [
            EXP_HOMO_HSPHERE_KERNEL,
            CONT_RELAX_KERNEL,
            HOMO_HSPHERE_KERNEL,
        ]:
            if len(self.options["theta0"]) == 1:
                self.options["theta0"] *= np.ones(d)
            else:
                raise ValueError(
                    "the length of theta0 (%s) should be equal to the number of dim (%s)."
                    % (len(self.options["theta0"]), d)
                )

        if self.options["use_het_noise"] and not self.options["eval_noise"]:
            if len(self.options["noise0"]) != self.nt:
                if len(self.options["noise0"]) == 1:
                    self.options["noise0"] *= np.ones(self.nt)
                else:
                    raise ValueError(
                        "for the heteroscedastic case, the length of noise0 (%s) should be equal to the number of observations (%s)."
                        % (len(self.options["noise0"]), self.nt)
                    )
        if not self.options["use_het_noise"]:
            if len(self.options["noise0"]) != 1:
                raise ValueError(
                    "for the homoscedastic noise case, the length of noise0 (%s) should be equal to one."
                    % (len(self.options["noise0"]))
                )

        if self.supports["training_derivatives"]:
            if not (1 in self.training_points[None]):
                raise Exception("Derivative values are needed for using the GEK model.")

    def _check_F(self, n_samples_F, p):
        """
        This function check the F-parameters of the model.
        """

        if n_samples_F != self.nt:
            raise Exception(
                "Number of rows in F and X do not match. Most "
                "likely something is going wrong with the "
                "regression model."
            )
        if p > n_samples_F:
            raise Exception(
                (
                    "Ordinary least squares problem is undetermined "
                    "n_samples=%d must be greater than the "
                    "regression model size p=%d."
                )
                % (self.nt, p)
            )


def compute_n_param(xtypes, cat_kernel, nx, d, n_comp, mat_dim):
    """
    Returns the he number of parameters needed for an homoscedastic or full group kernel.
    Parameters
     ----------
    xtypes: np.ndarray [dim]
            -the types (FLOAT,ORD,ENUM) of the input variables,
    cat_kernel : string
            -The kernel to use for categorical inputs. Only for non continuous Kriging,
    nx: int
            -The number of variables,
    d: int
            - n_comp or nx
    n_comp : int
            - if PLS, then it is the number of components else None,
    mat_dim : int
            - if PLS, then it is the number of components for matrix kernel (mixed integer) else None,
    Returns
    -------
     n_param: int
            - The number of parameters.
    """
    n_param = nx
    if n_comp is not None:
        n_param = d
        if cat_kernel == CONT_RELAX_KERNEL:
            return n_param
        if mat_dim is not None:
            return int(np.sum([l * (l - 1) / 2 for l in mat_dim]) + n_param)

    for i, xtyp in enumerate(xtypes):
        if isinstance(xtyp, tuple):
            if nx == d:
                n_param -= 1
            if cat_kernel in [EXP_HOMO_HSPHERE_KERNEL, HOMO_HSPHERE_KERNEL]:
                n_param += int(xtyp[1] * (xtyp[1] - 1) / 2)
            if cat_kernel == CONT_RELAX_KERNEL:
                n_param += int(xtyp[1])
    return n_param
