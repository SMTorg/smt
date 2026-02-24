"""
Reduced-likelihood evaluation for Kriging-based surrogate models.

This module encapsulates the computation of the reduced likelihood function,
its gradient, and its Hessian. It extracts what was previously inline in
:class:`KrgBased._reduced_likelihood_function` and related methods into
a self-contained class, following the Single Responsibility Principle.

Classes
-------
LikelihoodEvaluator
    Evaluates the reduced likelihood, gradient, and Hessian for standard
    Kriging models.
"""

import warnings

import numpy as np
from scipy import linalg


class LikelihoodEvaluator:
    """Evaluate the reduced likelihood for a standard Kriging model.

    This class owns the core linear-algebra computations
    (Cholesky, GLS, sigma2, log-likelihood) that were previously embedded
    in :meth:`KrgBased._reduced_likelihood_function` and its companion
    methods.

    Parameters
    ----------
    model : KrgBased
        The parent model.  The evaluator accesses ``model.options``,
        ``model.D``, ``model.F``, ``model.y_norma``, ``model.corr``, etc.
        through this reference.
    """

    def __init__(self, model):
        self._model = model

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _update_best_iteration(self, rlf_value, tmp_var):
        """Track the best likelihood seen so far (Cobyla failure recovery).

        ``best_iteration_fail`` and ``_thetaMemory`` are kept on the
        *model* for backward compatibility with subclasses (CCKRG,
        MFKPLSK, …) that access them directly.
        """
        m = self._model
        if (m.best_iteration_fail is not None) and (not np.isinf(rlf_value)):
            if rlf_value > m.best_iteration_fail:
                m.best_iteration_fail = rlf_value
                m._thetaMemory = np.array(tmp_var)
        elif (m.best_iteration_fail is None) and (not np.isinf(rlf_value)):
            m.best_iteration_fail = rlf_value
            m._thetaMemory = np.array(tmp_var)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, theta):
        """Compute the reduced likelihood function.

        This is the authoritative implementation for standard (non-sparse)
        Kriging.  Sparse GP models (SGP) override
        ``_reduced_likelihood_function`` on the model directly and never
        reach this evaluator.

        Parameters
        ----------
        theta : array-like
            Autocorrelation hyperparameters.

        Returns
        -------
        reduced_likelihood_function_value : float
            The reduced log-likelihood.
        par : dict
            Gaussian-process model parameters (``sigma2``, ``beta``,
            ``gamma``, ``C``, ``Ft``, ``Q``, ``G``, …).
        """
        from smt.surrogate_models.krg_based.krg_based import MixIntKernelType

        m = self._model

        reduced_likelihood_function_value = -np.inf
        par = {}

        # ---- nugget / noise ----
        nugget = m.options["nugget"]
        if m.options["eval_noise"]:
            if m.options["is_ri"]:
                nugget = 100.0 * np.finfo(np.double).eps
            else:
                nugget = 0
        noise = m.noise0
        tmp_var = theta
        if m.options["use_het_noise"]:
            noise = m.optimal_noise
        if m.options["eval_noise"] and not m.options["use_het_noise"]:
            theta = tmp_var[0:-1]
            noise = tmp_var[-1]

        # ---- compute correlation vector r ----
        if not m.is_continuous:
            dx = m.D
            if m.options["categorical_kernel"] == MixIntKernelType.CONT_RELAX:
                dx = m._get_cont_relax_dx(dx)
            try:
                r = m._matrix_data_corr(
                    corr=m.options["corr"],
                    design_space=m.design_space,
                    power=m.options["pow_exp_power"],
                    theta=theta,
                    theta_bounds=m.options["theta_bounds"],
                    dx=dx,
                    Lij=m.Lij,
                    n_levels=m.n_levels,
                    cat_features=m.cat_features,
                    cat_kernel=m.options["categorical_kernel"],
                    kplsk_second_loop=m.kplsk_second_loop,
                ).reshape(-1, 1)
                if np.isnan(r).any():
                    return reduced_likelihood_function_value, par
            except FloatingPointError:
                warnings.warn(
                    "Theta upper bound is too high.  please reduced it "
                    "in the parameter theta_bounds."
                )
                return reduced_likelihood_function_value, par
        else:
            m.corr.theta = theta
            try:
                r = m.corr(m.D).reshape(-1, 1)
                if np.isnan(r).any():
                    return reduced_likelihood_function_value, par
            except FloatingPointError:
                warnings.warn(
                    "Theta upper bound is too high.  please reduced it "
                    "in the parameter theta_bounds."
                )
                return reduced_likelihood_function_value, par

        # ---- build R matrix ----
        if m.options["is_ri"]:
            R_noisy = np.eye(m.nt) * (1.0 + nugget + noise)
            R_noisy[m.ij[:, 0], m.ij[:, 1]] = r[:, 0]
            R_noisy[m.ij[:, 1], m.ij[:, 0]] = r[:, 0]
            R = np.eye(m.nt) * (1.0 + nugget)
        else:
            R = np.eye(m.nt) * (1.0 + nugget + noise)

        R[m.ij[:, 0], m.ij[:, 1]] = r[:, 0]
        R[m.ij[:, 1], m.ij[:, 0]] = r[:, 0]

        p, q = m._get_pq()

        # ---- Cholesky decomposition ----
        C = None
        try:
            C = linalg.cholesky(R, lower=True)
        except (linalg.LinAlgError, ValueError) as e:
            print("exception : ", e)
            print(np.linalg.eig(R)[0])
            return reduced_likelihood_function_value, par

        # ---- reinterpolation path ----
        if m.options["is_ri"]:
            C_inv = np.linalg.inv(C)
            R_inv = np.dot(C_inv.T, C_inv)
            R_ri = R_noisy @ R_inv @ R_noisy
            par["C"] = C

            par["sigma2"] = None
            par["sigma2_ri"] = None
            _, _, sigma2_ri = self.compute_sigma2(
                R_ri, reduced_likelihood_function_value, par, p, q, is_ri=True
            )
            if sigma2_ri is not None:
                par["sigma2_ri"] = sigma2_ri * m.y_std**2.0

            reduced_likelihood_function_value, par, sigma2 = self.compute_sigma2(
                R_noisy, reduced_likelihood_function_value, par, p, q, is_ri=False
            )
            if sigma2 is not None:
                par["sigma2"] = sigma2 * m.y_std**2.0

            reduced_likelihood_function_value += m._reduced_log_prior(theta)
            self._update_best_iteration(reduced_likelihood_function_value, tmp_var)

            if reduced_likelihood_function_value > 1e15:
                reduced_likelihood_function_value = 1e15
            return reduced_likelihood_function_value, par

        # ---- standard GLS path ----
        Ft = linalg.solve_triangular(C, m.F, lower=True)
        Q, G = linalg.qr(Ft, mode="economic")
        sv = linalg.svd(G, compute_uv=False)
        rcondG = sv[-1] / sv[0]
        if rcondG < 1e-10:
            sv = linalg.svd(m.F, compute_uv=False)
            condF = sv[0] / sv[-1]
            if condF > 1e15:
                raise Exception(
                    "F is too ill conditioned. Poor combination "
                    "of regression model and observations."
                )
            else:
                return reduced_likelihood_function_value, par

        Yt = linalg.solve_triangular(C, m.y_norma, lower=True)
        beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
        rho = Yt - np.dot(Ft, beta)

        detR = (np.diag(C) ** (2.0 / m.nt)).prod()

        sigma2 = (rho**2.0).sum(axis=0) / (m.nt - p - q)
        reduced_likelihood_function_value = -(m.nt - p - q) * np.log10(
            sigma2.sum()
        ) - m.nt * np.log10(detR)
        par["sigma2"] = sigma2 * m.y_std**2.0
        par["beta"] = beta
        par["gamma"] = linalg.solve_triangular(C.T, rho)
        par["C"] = C
        par["Ft"] = Ft
        par["G"] = G
        par["Q"] = Q

        reduced_likelihood_function_value += m._reduced_log_prior(theta)
        self._update_best_iteration(reduced_likelihood_function_value, tmp_var)

        if reduced_likelihood_function_value > 1e15:
            reduced_likelihood_function_value = 1e15
        return reduced_likelihood_function_value, par

    # ------------------------------------------------------------------
    # Sigma2 for reinterpolation
    # ------------------------------------------------------------------

    def compute_sigma2(
        self, R, reduced_likelihood_function_value, par, p, q, is_ri=False
    ):
        """Compute the GP variance (sigma2) and update the likelihood.

        Used in the reinterpolation (``is_ri``) workflow.

        Parameters
        ----------
        R : np.ndarray
            Correlation matrix.
        reduced_likelihood_function_value : float
            Current reduced log-likelihood value.
        par : dict
            GP model parameters (updated in-place).
        p, q : int
            Regression / GP weight counts.
        is_ri : bool
            Whether this is the reinterpolation variance.

        Returns
        -------
        reduced_likelihood_function_value : float
        par : dict
        sigma2 : float or None
        """
        m = self._model

        try:
            C = linalg.cholesky(R, lower=True)
        except (linalg.LinAlgError, ValueError) as e:
            print("exception : ", e)
            print(np.linalg.eig(R)[0])
            sigma2 = par["sigma2"]
            if is_ri:
                sigma2 = par["sigma2_ri"]
            return reduced_likelihood_function_value, par, sigma2

        Ft = linalg.solve_triangular(C, m.F, lower=True)
        Q, G = linalg.qr(Ft, mode="economic")
        sv = linalg.svd(G, compute_uv=False)
        rcondG = sv[-1] / sv[0]
        if rcondG < 1e-10:
            sv = linalg.svd(m.F, compute_uv=False)
            condF = sv[0] / sv[-1]
            if condF > 1e15:
                raise Exception(
                    "F is too ill conditioned. Poor combination "
                    "of regression model and observations."
                )
            else:
                return reduced_likelihood_function_value, par, None

        Yt = linalg.solve_triangular(C, m.y_norma, lower=True)
        beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
        rho = Yt - np.dot(Ft, beta)

        detR = (np.diag(C) ** (2.0 / m.nt)).prod()

        sigma2 = (rho**2.0).sum(axis=0) / (m.nt - p - q)
        reduced_likelihood_function_value = -(m.nt - p - q) * np.log10(
            sigma2.sum()
        ) - m.nt * np.log10(detR)

        if not is_ri:
            par["beta"] = beta
            par["gamma"] = linalg.solve_triangular(C.T, rho)
            par["Ft"] = Ft
            par["G"] = G
            par["Q"] = Q
            par["C_noisy"] = C

        return reduced_likelihood_function_value, par, sigma2

    # ------------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------------

    def gradient(self, theta):
        """Evaluate the gradient of the reduced likelihood.

        Parameters
        ----------
        theta : array-like
            Autocorrelation hyperparameters.

        Returns
        -------
        grad_red : np.ndarray (dim, 1)
            Gradient of the reduced likelihood.
        par : dict
            GP model parameters, augmented with derivative info.
        """
        m = self._model
        grad_red = 1
        par = {}
        red, par = self.evaluate(theta)
        try:
            C = par["C"]
            gamma = par["gamma"]
            Q = par["Q"]
            G = par["G"]
            sigma_2 = par["sigma2"] + m.options["nugget"]
        except KeyError:
            return grad_red, par

        C = par["C"]
        gamma = par["gamma"]
        Q = par["Q"]
        G = par["G"]
        sigma_2 = par["sigma2"] + m.options["nugget"]

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
            dr = m.corr(m.D, grad_ind=i_der)
            dr_all.append(dr)

            dR = np.zeros((m.nt, m.nt))
            dR[m.ij[:, 0], m.ij[:, 1]] = dr[:, 0]
            dR[m.ij[:, 1], m.ij[:, 0]] = dr[:, 0]

            # Compute beta derivatives
            Cinv_dR_gamma = linalg.solve_triangular(C, np.dot(dR, gamma), lower=True)
            dbeta = -linalg.solve_triangular(G, np.dot(Q.T, Cinv_dR_gamma))
            arg_all.append(Cinv_dR_gamma)

            dbeta_all.append(dbeta)

            # Compute mu derivatives
            dmu = np.dot(m.F, dbeta)
            dmu_all.append(dmu)

            # Compute log(detR) derivatives
            tr_1 = linalg.solve_triangular(C, dR, lower=True)
            tr = linalg.solve_triangular(C.T, tr_1)
            tr_all.append(tr)

            # Compute Sigma2 Derivatives
            dsigma_2 = (
                (1 / m.nt)
                * (
                    -dmu.T.dot(gamma)
                    - gamma.T.dot(dmu)
                    - np.dot(gamma.T, dR.dot(gamma))
                )
                * m.y_std**2.0
            )
            dsigma_all.append(dsigma_2)

            # Compute reduced log likelihood derivatives
            grad_red[i_der] = (
                -m.nt / np.log(10) * (dsigma_2 / sigma_2 + np.trace(tr) / m.nt)
            ).item()

        par["dr"] = dr_all
        par["tr"] = tr_all
        par["dmu"] = dmu_all
        par["arg"] = arg_all
        par["dsigma"] = dsigma_all
        par["dbeta_all"] = dbeta_all

        grad_red = np.atleast_2d(grad_red).T

        grad_red += m._reduced_log_prior(theta, grad=True)
        return grad_red, par

    # ------------------------------------------------------------------
    # Hessian
    # ------------------------------------------------------------------

    def hessian(self, theta):
        """Evaluate the Hessian of the reduced likelihood.

        Parameters
        ----------
        theta : array-like
            Autocorrelation hyperparameters.

        Returns
        -------
        hess : np.ndarray
            Hessian values (upper triangle, flattened).
        hess_ij : np.ndarray
            Index pairs for the Hessian entries.
        par : dict
            GP model parameters, augmented with derivative info.
        """
        m = self._model
        dred, par = self.gradient(theta)

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
        log_prior = m._reduced_log_prior(theta, hessian=True)

        for omega in range(nb_theta):
            ind_0 = ind_1
            ind_1 = ind_0 + nb_theta - omega
            hess_ij[ind_0:ind_1, 0] = omega
            hess_ij[ind_0:ind_1, 1] = np.arange(omega, nb_theta)

            dRdomega = np.zeros((m.nt, m.nt))
            dRdomega[m.ij[:, 0], m.ij[:, 1]] = dr_all[omega][:, 0]
            dRdomega[m.ij[:, 1], m.ij[:, 0]] = dr_all[omega][:, 0]

            dmudomega = dmu_all[omega]
            Cinv_dmudomega = linalg.solve_triangular(C, dmudomega, lower=True)
            Rinv_dmudomega = linalg.solve_triangular(C.T, Cinv_dmudomega)
            Rinv_dmudomega_all.append(Rinv_dmudomega)
            Rinv_dRdomega_gamma = linalg.solve_triangular(C.T, arg_all[omega])
            Rinv_dRdomega_gamma_all.append(Rinv_dRdomega_gamma)

            for i, eta in enumerate(hess_ij[ind_0:ind_1, 1]):
                dRdeta = np.zeros((m.nt, m.nt))
                dRdeta[m.ij[:, 0], m.ij[:, 1]] = dr_all[eta][:, 0]
                dRdeta[m.ij[:, 1], m.ij[:, 0]] = dr_all[eta][:, 0]
                dr_eta_omega = m.corr(m.D, grad_ind=omega, hess_ind=eta)
                dRdetadomega = np.zeros((m.nt, m.nt))
                dRdetadomega[m.ij[:, 0], m.ij[:, 1]] = dr_eta_omega[:, 0]
                dRdetadomega[m.ij[:, 1], m.ij[:, 0]] = dr_eta_omega[:, 0]

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
                dmudetadomega = np.dot(m.F, dbetadetadomega)

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
                    (1 / m.nt)
                    * (sigma_arg_1 + sigma_arg_2 + sigma_arg_3 + sigma_arg_4)
                    * m.y_std**2.0
                )

                # Compute Hessian
                dreddetadomega_tr_1 = np.trace(np.dot(tr_all[eta], tr_all[omega]))

                dreddetadomega_tr_2 = np.trace(
                    linalg.solve_triangular(
                        C.T, linalg.solve_triangular(C, dRdetadomega, lower=True)
                    )
                )

                dreddetadomega_arg1 = (m.nt / sigma_2) * (
                    dsigma2detadomega - (1 / sigma_2) * dsigma[omega] * dsigma[eta]
                )
                dreddetadomega = (
                    -(dreddetadomega_arg1 - dreddetadomega_tr_1 + dreddetadomega_tr_2)
                    / m.nt
                )

                hess[ind_0 + i, 0] = (m.nt / np.log(10) * dreddetadomega).item()

                if eta == omega:
                    hess[ind_0 + i, 0] += log_prior[eta].item()
            par["Rinv_dR_gamma"] = Rinv_dRdomega_gamma_all
            par["Rinv_dmu"] = Rinv_dmudomega_all
        return hess, hess_ij, par
