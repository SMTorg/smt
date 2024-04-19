"""
Author: Hugo Reimeringer <hugo.reimeringer@onera.fr>
"""

from sklearn.decomposition import PCA
import numpy as np

from smt.applications.application import SurrogateBasedApplication
from smt.surrogate_models import KRG
from smt.utils.checks import ensure_2d_array

PODI_available_models = {"KRG": lambda: KRG()}

# merge set_training_values and train into fit ?


class PODI(SurrogateBasedApplication):
    """
    Class for Proper Orthogonal Decomposition and Interpolation surrogate models based.

    Attributes
    ----------
    n_modes : int
        Number of kept modes during the POD.
    left_basis : np.ndarray
        Left basis of the POD.
    singular_values : np.ndarray
        Singular values of the POD
    interp_list : list[SurrogateModel]
        List containing the kriging models used.
    training_values_set : bool
        Indicates if the training values have already been set.
    train_done : bool
        Indicates if the training has been performed yet.

    Example
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

        self.n_modes = None
        self.left_basis = None
        self.singular_values = None

        self.pod_computed = False
        self.interp_options_set = False
        self.training_values_set = False
        self.train_done = False

        self.interp_coeff = None

    @staticmethod
    def choice_n_modes_tol(EV_list: np.ndarray, tol: float) -> (int, float):
        """
        Calculates the required number of kept modes to explain the intended ratio of variance.

        Parameters
        ----------
        EV_list : np.ndarray
            List of the explained variance of each mode in the POD.
        tol : float
            Desired tolerance for the POD.

        Returns
        -------
        n_modes : int
            Kept modes according to the tolerance.
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

    def compute_pod(
        self,
        database: np.ndarray,
        tol: float = None,
        n_modes: int = None,
        random_state: int = None,
    ) -> None:
        """
        Performs the POD

        Parameters
        ----------
        database : np.ndarray[nt, ny]
            Snapshot matrix. Each row correspond to a snapshot.
        tol : float
            Desired tolerance for the pod (if n_modes not set).
        n_modes : int
            Desired number of kept modes for the pod (if tol not set).
        random_state : int
            Numpy RandomState object or seed number which controls random draws for internal optim. (optional)

        Example
        ----------
        sm.compute_pod(database, tol = 0.99)
        """
        choice_svd = None

        svd = PCA(svd_solver="randomized", random_state=random_state)

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

        database = ensure_2d_array(database, "database")

        self.n_snapshot = database.shape[0]
        self.ny = database.shape[1]

        svd.fit(database)
        self.left_basis = svd.components_.T
        self.singular_values = svd.singular_values_
        EV_list = svd.explained_variance_

        if choice_svd == "tol":
            self.n_modes, self.EV_ratio = PODI.choice_n_modes_tol(EV_list, tol)
        else:
            if self.n_modes > self.n_snapshot:
                raise ValueError(
                    "the number of kept modes can't be superior to the number of data values (snapshots)"
                )
            self.EV_ratio = sum(EV_list[: self.n_modes]) / sum(EV_list) * 100

        self.mean = np.atleast_2d(database.T.mean(axis=1)).T
        self.basis = np.array(self.left_basis[:, : self.n_modes])
        self.coeff = np.dot(self.basis.T, database.T - self.mean).T

        # self.interp_coeff = []
        # for i in range(self.n_modes):
        #     self.interp_coeff.append(KRG(print_global=False))

        self.pod_computed = True
        self.interp_options_set = False
        self.training_values_set = False

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
        self, interp_type: str = "KRG", interp_options_list: list = [{}]
    ) -> None:
        """
        Set the options for the interpolation surrogate models used.

        Parameters
        ----------
        interp_type : str
            Name of the type of surrogate model that will be used for the whole set.
            By default, the Kriging model is used (KRG).

        interp_options_list : list[dict]
            List containing dictionnaries for the options.
            The k-th dictionnary corresponds to the options of the k-th interpolation model.
            If the options are common to all surogate models, only a single dictionnary is required in the list.
            The available options can be found in the documentation of the corresponding surrogate models.
            By default, the print_global options are set to 'False'.
            
        Example
        --------
        >>> interp_type = "KRG"
        >>> dict1 = {'corr' : 'matern52', 'theta0' : [1e-2]}
        >>> dict2 = {'poly' : 'quadratic'}
        >>> interp_options_list = [dict1, dict2]
        >>> sm.set_interp_options(interp_type, interp_options_list)
        """

        if not self.pod_computed:
            raise RuntimeError(
                "'POD' method must have been succesfully executed before trying to set the interpolation options."
            )

        if interp_type not in PODI_available_models.keys():
            raise ValueError(
                f"the surrogate model type should be one of the following : {', '.join(self.available_models_name)}"
            )

        if interp_options_list is None:
            raise ValueError(
                "the interp_options_list should be indicated when executing 'set_interp_options' method."
            )

        if len(interp_options_list) == self.n_modes:
            mode_options = "local"
        elif len(interp_options_list) == 1:
            mode_options = "global"
        else:
            raise ValueError(
                f"expected interp_options_list of size {self.n_modes} or 1, but got {len(interp_options_list)}."
            )

        self.interp_coeff = []
        for i in range(self.n_modes):
            if mode_options == "local":
                index = i
            elif mode_options == "global":
                index = 0

            sm_i = PODI_available_models[interp_type]()
            sm_i.options["print_global"] = False

            for key in interp_options_list[index].keys():
                sm_i.options[key] = interp_options_list[index][key]
            
            self.interp_coeff.append(sm_i)

        self.interp_options_set = True
        self.training_values_set = False

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

        if not self.pod_computed:
            raise RuntimeError(
                "'POD' method must have been succesfully executed before trying to set the training values."
            )
        if not self.interp_options_set:
            raise RuntimeError(
                "the surrogate models' options should have been set before trying to set the training values."
            )

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
                "'POD' method must have been succesfully executed before trying to train the models."
            )
        if not self.interp_options_set:
            raise RuntimeError(
                "the surrogate models' options should have been successfully set before trying to train the models."
            )
        if not self.training_values_set:
            raise RuntimeError(
                "the training values should have been set before trying to train the models."
            )

        for interp_coeff in self.interp_coeff:
            interp_coeff.train()

        self.train_done = True

    def get_interp_coef(self) -> np.ndarray:
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

        self.dim_new = xn.shape[1]

        if self.dim_new != self.nx:
            raise ValueError(
                f"the data values and the new values must be the same size, {self.dim_new} != {self.nx}"
            )

        self.n_new = xn.shape[0]
        mean_coeff_interp = np.zeros((self.n_new, self.n_modes))

        for i in range(self.n_modes):
            mu_i = self.interp_coeff[i].predict_values(xn)
            mean_coeff_interp[:, i] = mu_i[:, 0]

        y = self.mean.T + np.dot(mean_coeff_interp, self.basis.T)

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

        dim_new = xn.shape[1]

        if dim_new != self.nx:
            raise ValueError(
                f"the data values and the new values must be the same size, {self.dim_new} != {self.nx}"
            )

        self.n_new = xn.shape[0]
        var_coeff_interp = np.zeros((self.n_new, self.n_modes))

        for i in range(self.n_modes):
            sigma_i_square = self.interp_coeff[i].predict_variances(xn)
            var_coeff_interp[:, i] = sigma_i_square[:, 0]

        s2 = np.dot(var_coeff_interp, (self.basis**2).T)

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

        dim_new = xn.shape[1]

        if dim_new != self.nx:
            raise ValueError(
                f"the data values and the new values must be the same size, {self.dim_new} != {self.nx}"
            )

        n_new = xn.shape[0]
        deriv_coeff_interp = np.zeros((n_new, self.n_modes))

        for i in range(self.n_modes):
            deriv_coeff_interp[:, i] = self.interp_coeff[i].predict_derivatives(xn, d)[:, 0]

        dy_dx = np.dot(deriv_coeff_interp, self.basis.T)

        return dy_dx
