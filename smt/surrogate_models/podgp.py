"""
Author: Hugo Reimeringer <hugo.reimeringer@onera.fr>
"""

from sklearn.decomposition import PCA
import numpy as np

from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.surrogate_models import KRG
from smt.utils.checks import ensure_2d_array


class PODGP(SurrogateModel):
    """
    Class for Proper Orthogonal Decomposition and Gaussian Processes (PODGP) based surrogate model.

    Attributes
    ----------
    n_mods : int
        Number of kept mods during the POD.
    left_basis : np.ndarray
        Left basis of the POD.
    singular_values : np.ndarray
        Singular values of the POD
    sm_list : list[SurrogateModel]
        List containing the kriging models used.
    training_values_set : bool
        Indicates if the training values have already been set.
    train_done : bool
        Indicates if the training has been performed yet.

    Example
    --------
    >>> from smt.surrogate_models import PODGP
    >>> sm = PODGP(print_training=False)
    >>> sm.options['print_prediction'] = False
    """

    name = "PODGP"

    def _initialize(self) -> None:
        super()._initialize()
        supports = self.supports

        self.n_mods = None
        self.left_basis = None
        self.singular_values = None
        self.training_values_set = False
        self.train_done = False
        self.sm_list = None

        supports["variances"] = True
        supports["derivatives"] = True

    @staticmethod
    def choice_n_mods_tol(EV_list: np.ndarray, tol: float) -> (int, float):
        """
        Calculates the required number of kept mods to explain the intended ratio of variance.

        Parameters
        ----------
        EV_list : np.ndarray
            List of the explained variance of each mod in the POD.
        tol : float
            Desired tolerance for the POD.

        Returns
        -------
        n_mods : int
            Kept mods according to the tolerance.
        EV_ratio : float
            Actual ratio of explained variance.
        """

        sum_tot = sum(EV_list)
        sum_ = 0
        for i in range(len(EV_list)):
            sum_ += EV_list[i]
            EV_ratio = sum_ / sum_tot
            if sum_ / sum_tot >= tol:
                return i + 1, EV_ratio

    def POD(
        self,
        database: np.ndarray,
        tol: float = None,
        n_mods: int = None,
        random_state: int = None,
    ) -> None:
        """
        Performs the POD

        Parameters
        ----------
        database : np.ndarray[nt, ny]
            Snapshot matrix. Each column correspond to a snapshot.
        tol : float
            Desired tolerance for the pod (if n_mods not set).
        n_mods : int
            Desired number of kept mod for the pod (if tol not set).
        random_state : int
            Numpy RandomState object or seed number which controls random draws for internal optim. (optional)

        Example
        ----------
        sm.POD(database, tol = 0.99)
        """
        choice_svd = None

        svd = PCA(svd_solver="randomized", random_state=random_state)

        if n_mods is not None:
            self.n_mods = n_mods
            choice_svd = "mod"

        if tol is not None:
            if choice_svd is not None:
                raise ValueError(
                    "pod can't use both arguments 'n_mods' and 'tol' at the same time"
                )
            else:
                choice_svd = "tol"

        if choice_svd is None:
            raise ValueError(
                "either one of the arguments 'n_mods' and 'tol' must be specified"
            )

        database = ensure_2d_array(database, "database")

        self.n_snapshot = database.shape[0]
        self.ny = database.shape[1]

        svd.fit(database)
        self.left_basis = svd.components_.T
        self.singular_values = svd.singular_values_
        EV_list = svd.explained_variance_

        if choice_svd == "tol":
            self.n_mods, self.EV_ratio = PODGP.choice_n_mods_tol(EV_list, tol)
        else:
            if self.n_mods > self.n_snapshot:
                raise ValueError(
                    "the number of kept mods can't be superior to the number of data values (snapshots)"
                )
            self.EV_ratio = sum(EV_list[: self.n_mods]) / sum(EV_list) * 100

        self.mean = np.atleast_2d(database.T.mean(axis=1)).T
        self.basis = np.array(self.left_basis[:, : self.n_mods])
        self.coeff = np.dot(self.basis.T, database.T - self.mean).T

        self.training_values_set = False
        self.train_done = False

        self.sm_list = []
        for i in range(self.n_mods):
            self.sm_list.append(KRG(print_global=False))

    def get_left_basis(self) -> np.ndarray:
        """
        Getter for the left basis of the POD.

        Returns
        -------
        left_basis : np.ndarray
            Left basis of the POD.
        """
        return self.left_basis

    def get_singular_values(self) -> np.ndarray:
        """
        Getter for the singular values from the Sigma matrix of the POD.

        Returns
        -------
        singular_values : np.ndarray
            Singular values of the POD.
        """
        return self.singular_values

    def get_ev_ratio(self) -> float:
        """
        Getter for the explained variance ratio with the kept mods.

        Returns
        -------
        EV_ratio : float
            Explained variance ratio with the current kept mods.
        """
        return self.EV_ratio

    def get_n_mods(self) -> int:
        """
        Getter for the number of mods kept during the POD.

        Returns
        -------
        n_mods : int
            number of mods kept during the POD.
        """
        return self.n_mods

    def set_GP_options(self, GP_options_list: list = [{}]) -> None:
        """
        Set the options for the GP surrogate models used.

        Parameters
        ----------
        GP_options_list : list[dict]
            List containing dictionnaries for the options (optional parameter).
            The k-th dictionnary corresponds to the options of the k-th GP model.
            If the options are common to all surogate models, only a single dictionnary is required in the list.
            The available options are the same as the kriging one's.

        Example
        --------
        >>> dict1 = {'corr' : 'matern52', 'theta0' : [1e-2]}
        >>> dict2 = {'poly' : 'quadratic'}
        >>> GP_options_list = [dict1, dict2]
        >>> sm.set_GP_options(GP_options_list)
        """

        if self.sm_list is None:
            raise RuntimeError(
                "'POD' method must have been succesfully executed before trying to set the GP options."
            )
        if len(GP_options_list) == 1:
            mod_options = "global"
        elif len(GP_options_list) != self.n_mods:
            raise ValueError(
                f"expected GP_options_list of size n_mods = {self.n_mods}, but got {len(GP_options_list)} instead."
            )
        else:
            mod_options = "local"

        for i in range(self.n_mods):
            if mod_options == "local":
                index = i
            elif mod_options == "global":
                index = 0
            for key in GP_options_list[index].keys():
                self.sm_list[i].options[key] = GP_options_list[index][key]

    def set_training_values(self, xt: np.ndarray, name: str = None) -> None:
        """
        Set training data (values).

        Parameters
        ----------
        xt : np.ndarray[nt, nx]
            The input values for the nt training points.
        name : str
            An optional label for the group of training points being set.
            This is only used in special situations (e.g., multi-fidelity applications).
        """

        xt = ensure_2d_array(xt, "xt")

        if self.sm_list is None:
            raise RuntimeError(
                "'POD' method must have been succesfully executed before trying to set the training values."
            )
        self.nt = xt.shape[0]
        self.nx = xt.shape[1]

        if self.nt != self.n_snapshot:
            raise ValueError(
                f"there must be the same amount of train values than data values, {self.nt} != {self.n_snapshot}."
            )

        for i in range(self.n_mods):
            self.sm_list[i].set_training_values(xt, self.coeff[:, i])
            self.training_points[name][i] = [xt, self.coeff[:, i]]
        self.training_values_set = True

    # def train(self) -> None:
    #     """
    #     Performs the training of the model.
    #     """
    #     if not self.training_values_set:
    #         raise RuntimeError(
    #             "the training values should have been set before trying to train the model"
    #         )

    #     for i in range(self.n_mods):
    #         super().train(self.sm_list[i])
    #     self.train_done = True

    #     return None

    def _train(self) -> None:
        """
        Performs the training of the model.
        """

        if not self.training_values_set:
            raise RuntimeError(
                "the training values should have been set before trying to train the model"
            )

        for i in range(self.n_mods):
            self.sm_list[i].train()
        self.train_done = True

    def get_gp_coef(self) -> np.ndarray:
        """
        Getter for the list of the GP surrogate models used

        Returns
        -------
        sm_list : np.ndarray[n_mods]
            List of the kriging models used for the POD coefficients.
        """
        return self.sm_list

    def _predict_values(self, xn) -> np.ndarray:
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
                "the model should have been trained before trying to make a prediction"
            )

        self.dim_new = xn.shape[1]

        if self.dim_new != self.nx:
            raise ValueError(
                f"the data values and the new values must be the same size, {self.dim_new} != {self.nx}"
            )

        self.n_new = xn.shape[0]
        mean_coeff_gp = np.zeros((self.n_new, self.n_mods))

        for i in range(self.n_mods):
            mu_i = self.sm_list[i].predict_values(xn)
            mean_coeff_gp[:, i] = mu_i[:, 0]

        y = self.mean.T + np.dot(mean_coeff_gp, self.basis.T)

        return y

    def _predict_variances(self, xn) -> np.ndarray:
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

        dim_new = xn.shape[1]

        if dim_new != self.nx:
            raise ValueError(
                f"the data values and the new values must be the same size, {self.dim_new} != {self.nx}"
            )

        self.n_new = xn.shape[0]
        var_coeff_gp = np.zeros((self.n_new, self.n_mods))

        for i in range(self.n_mods):
            sigma_i_square = self.sm_list[i].predict_variances(xn)
            var_coeff_gp[:, i] = sigma_i_square[:, 0]

        s2 = np.dot(var_coeff_gp, (self.basis**2).T)

        return s2

    def _predict_derivatives(self, xn, kx) -> np.ndarray:
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

        dim_new = xn.shape[1]

        if dim_new != self.nx:
            raise ValueError(
                f"the data values and the new values must be the same size, {self.dim_new} != {self.nx}"
            )

        n_new = xn.shape[0]
        deriv_coeff_gp = np.zeros((n_new, self.n_mods))

        for i in range(self.n_mods):
            deriv_coeff_gp[:, i] = self.sm_list[i].predict_derivatives(xn, d)[:, 0]

        dy_dx = np.dot(deriv_coeff_gp, self.basis.T)

        return dy_dx
