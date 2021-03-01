"""
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.

Mixture of Experts
"""
# TODO : support for best number of clusters
# TODO : implement verbosity 'print_global'
# TODO : documentation

import numpy as np
import warnings

OLD_SKLEARN = False
try:  # scikit-learn < 0.20.0
    from sklearn.mixture import GMM as GaussianMixture

    OLD_SKLEARN = True
except:
    from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

from smt.utils.options_dictionary import OptionsDictionary
from smt.applications.application import SurrogateBasedApplication
from smt.utils.misc import compute_rms_error
from smt.surrogate_models.surrogate_model import SurrogateModel

warnings.filterwarnings("ignore", category=DeprecationWarning)

MOE_EXPERT_NAMES = [
    "KRG",
    "KPLS",
    "KPLSK",
    "LS",
    "QP",
    "RBF",
    "IDW",
    "RMTB",
    "RMTC",
]


class MOESurrogateModel(SurrogateModel):
    """Wrapper class exposing MOE features as a SurrogateModel subclass."""

    name = "MOE"

    def _initialize(self):
        super(MOESurrogateModel, self)._initialize()

        # Copy over options from MOE object
        self.moe = moe = MOE()
        for key, data in moe.options._declared_entries.items():
            self.options._declared_entries[key] = data

            value = moe.options[key]
            if value is not None:
                self.options[key] = value

    def _setup(self):
        for key in self.moe.options._declared_entries:
            if key in self.options:
                self.moe.options[key] = self.options[key]

        # self.supports['derivatives'] = self.options['derivatives_support']  # Interface not yet implemented
        self.supports["variances"] = self.options["variances_support"]

    def train(self):
        if len(self.training_points) == 0:
            xt = self.options["xt"]
            yt = self.options["yt"]
            self.set_training_values(xt, yt)

        super(MOESurrogateModel, self).train()

    def _train(self):
        self._setup()

        for name in self.training_points:
            xt, yt = self.training_points[name][0]
            self.moe.set_training_values(xt, yt, name=name)
        self.moe.train()

    def _predict_values(self, x: np.ndarray) -> np.ndarray:
        return self.moe.predict_values(x)

    def _predict_variances(self, x: np.ndarray) -> np.ndarray:
        return self.moe.predict_variances(x)


class MOE(SurrogateBasedApplication):

    # Names of experts available to be part of the mixture
    AVAILABLE_EXPERTS = [
        name
        for name in MOE_EXPERT_NAMES
        if name in SurrogateBasedApplication._surrogate_type
    ]

    def _initialize(self):
        super(MOE, self)._initialize()
        declare = self.options.declare

        declare("xt", None, types=np.ndarray, desc="Training inputs")
        declare("yt", None, types=np.ndarray, desc="Training outputs")
        declare(
            "ct",
            None,
            types=np.ndarray,
            desc="Training derivative outputs used for clustering",
        )

        declare("xtest", None, types=np.ndarray, desc="Test inputs")
        declare("ytest", None, types=np.ndarray, desc="Test outputs")

        declare("n_clusters", 2, types=int, desc="Number of clusters")
        declare(
            "smooth_recombination",
            True,
            types=bool,
            desc="Continuous cluster transition",
        )
        declare(
            "heaviside_optimization",
            False,
            types=bool,
            desc="Optimize Heaviside scaling factor when smooth recombination is used",
        )

        declare(
            "derivatives_support",
            False,
            types=bool,
            desc="Use only experts that support derivatives prediction",
        )
        declare(
            "variances_support",
            False,
            types=bool,
            desc="Use only experts that support variance prediction",
        )
        declare(
            "allow",
            [],
            desc="Names of allowed experts to be possibly part of the mixture. "
            "Empty list corresponds to all surrogates allowed.",
        )
        declare(
            "deny",
            [],
            desc="Names of forbidden experts",
        )

        self.x = None
        self.y = None
        self.c = None

        self.n_clusters = None
        self.smooth_recombination = None
        self.heaviside_optimization = None
        self.heaviside_factor = 1.0

        # dictionary {name: class} of possible experts wrt to options
        self._enabled_expert_types = self._get_enabled_expert_types()

        # list of experts after MOE training
        self._experts = []

        self.xt = None
        self.yt = None

    @property
    def enabled_experts(self):
        """
        Returns the names of enabled experts after taking into account MOE options
        """
        self._enabled_expert_types = self._get_enabled_expert_types()
        return list(self._enabled_expert_types.keys())

    def set_training_values(self, xt, yt, name=None):
        """
        Set training data (values).

        Parameters
        ----------
        xt : np.ndarray[nt, nx] or np.ndarray[nt]
            The input values for the nt training points.
        yt : np.ndarray[nt, ny] or np.ndarray[nt]
            The output values for the nt training points.
        name : str or None
            An optional label for the group of training points being set.
            This is only used in special situations (e.g., multi-fidelity applications).
        """
        self.xt = xt
        self.yt = yt

    def train(self):
        """
        Supports for surrogate model API.
        Build and train the mixture of experts surrogate.
        """
        if self.xt is not None and self.yt is not None:
            # set_training_values has been called
            self.x = x = self.xt
            self.y = y = self.yt
        else:
            self.x = x = self.options["xt"]
            self.y = y = self.options["yt"]
        self.c = c = self.options["ct"]
        if not self.c:
            self.c = c = y

        self.n_clusters = self.options["n_clusters"]
        self.smooth_recombination = self.options["smooth_recombination"]
        self.heaviside_optimization = (
            self.options["smooth_recombination"]
            and self.options["heaviside_optimization"]
        )
        self.heaviside_factor = 1.0

        self._check_inputs()

        self._enabled_expert_types = self._get_enabled_expert_types()
        self._experts = []

        # Set test values and trained values
        xtest = self.options["xtest"]
        ytest = self.options["ytest"]
        values = np.c_[x, y, c]
        test_data_present = xtest is not None and ytest is not None
        if test_data_present:
            self.test_values = np.c_[xtest, ytest]
            self.training_values = values
        else:
            self.test_values, self.training_values = self._extract_part(values, 10)

        self.ndim = nx = x.shape[1]
        xt = self.training_values[:, 0:nx]
        yt = self.training_values[:, nx : nx + 1]
        ct = self.training_values[:, nx + 1 :]

        # Clustering
        self.cluster = GaussianMixture(
            n_components=self.n_clusters, covariance_type="full", n_init=20
        )
        self.cluster.fit(np.c_[xt, ct])
        if not self.cluster.converged_:
            raise Exception("Clustering not converged")

        # Choice of the experts and training
        self._fit(xt, yt, ct)

        xtest = self.test_values[:, 0:nx]
        ytest = self.test_values[:, nx : nx + 1]
        # Heaviside factor
        if self.heaviside_optimization and self.n_clusters > 1:
            self.heaviside_factor = self._find_best_heaviside_factor(xtest, ytest)
            print("Best Heaviside factor = {}".format(self.heaviside_factor))
            self.distribs = self._create_clusters_distributions(self.heaviside_factor)

        if not test_data_present:
            # if we have used part of data to validate, fit on overall data
            self._fit(x, y, c, new_model=False)

    def predict_values(self, x):
        """
        Predict the output values at a set of points.

        Parameters
        ----------
        x : np.ndarray[nt, nx] or np.ndarray[nt]
            Input values for the prediction points.

        Returns
        -------
        y : np.ndarray[nt, ny]
            Output values at the prediction points.
        """
        if self.smooth_recombination:
            y = self._predict_smooth_output(x)
        else:
            y = self._predict_hard_output(x)
        return y

    def predict_variances(self, x):
        """
        Predict the output variances at a set of points.

        Parameters
        ----------
        x : np.ndarray[nt, nx] or np.ndarray[nt]
            Input values for the prediction points.

        Returns
        -------
        y : np.ndarray[nt, ny]
            Output variances at the prediction points.
        """

        if not self.options["variances_support"]:
            raise RuntimeError(
                "Experts not selected taking variance support into account: use variances_support=True "
                "when creating MOE"
            )

        if self.smooth_recombination:
            y = self._predict_smooth_output(x, output_variances=True)
        else:
            y = self._predict_hard_output(x, output_variances=True)
        return y

    def _check_inputs(self):
        """
        Check the input data given by the client is correct.
        raise Value error with relevant message
        """
        if self.x is None or self.y is None:
            raise ValueError("check x and y values")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(
                "The number of input points %d doesn t match with the number of output points %d."
                % (self.x.shape[0], self.y.shape[0])
            )
        if self.y.shape[0] != self.c.shape[0]:
            raise ValueError(
                "The number of output points %d doesn t match with the number of criterion weights %d."
                % (self.y.shape[0], self.c.shape[0])
            )
        # choice of number of cluster
        max_n_clusters = int(len(self.x) / 10) + 1
        if self.n_clusters > max_n_clusters:
            print("Number of clusters should be inferior to {0}".format(max_n_clusters))
            raise ValueError(
                "The number of clusters is too high considering the number of points"
            )

    def _get_enabled_expert_types(self):
        """
        Select relevant surrogate models (experts) regarding MOE feature options
        """
        prototypes = {
            name: smclass()
            for name, smclass in self._surrogate_type.items()
            if name in MOE_EXPERT_NAMES
        }
        if self.options["derivatives_support"]:
            prototypes = {
                name: proto
                for name, proto in prototypes.items()
                if proto.supports["derivatives"]
            }
        if self.options["variances_support"]:
            prototypes = {
                name: proto
                for name, proto in prototypes.items()
                if proto.supports["variances"]
            }
        if self.options["allow"]:
            prototypes = {
                name: proto
                for name, proto in prototypes.items()
                if name in self.options["allow"]
            }
        if self.options["deny"]:
            prototypes = {
                name: proto
                for name, proto in prototypes.items()
                if name not in self.options["deny"]
            }
        if not prototypes:
            ValueError(
                "List of possible experts is empty: check support, allow and deny options wrt"
            )
        return {name: self._surrogate_type[name] for name in prototypes}

    def _fit(self, x_trained, y_trained, c_trained, new_model=True):
        """
        Find the best model for each cluster (clustering already done) and train it if new_model is True
        otherwise train the points given (choice of best models by cluster already done)

        Arguments
        ---------
        - x_trained: array_like
            Input training samples
        - y_trained: array_like
            Output training samples
        - c_trained: array_like
            Clustering training samples
        - new_model : bool (optional)
            Set true to search the best local model

        """
        self.distribs = self._create_clusters_distributions(self.heaviside_factor)

        cluster_classifier = self.cluster.predict(np.c_[x_trained, c_trained])

        # sort trained_values for each cluster
        clusters = self._cluster_values(np.c_[x_trained, y_trained], cluster_classifier)

        # find model for each cluster
        for i in range(self.n_clusters):
            if new_model:
                model = self._find_best_model(clusters[i])
                self._experts.append(model)
            else:  # retrain the experts with the
                trained_values = np.array(clusters[i])
                x_trained = trained_values[:, 0 : self.ndim]
                y_trained = trained_values[:, self.ndim]
                self._experts[i].set_training_values(x_trained, y_trained)
                self._experts[i].train()

    def _predict_hard_output(self, x, output_variances=False):
        """
        This method predicts the output of a x samples for a
        discontinuous recombination.

        Arguments
        ---------
        - x : array_like
            x samples

        Return
        ------
        - predicted_values : array_like
            predicted output

        """
        predicted_values = []
        probs = self._proba_cluster(x)
        sort_cluster = np.apply_along_axis(np.argmax, 1, probs)

        for i in range(len(sort_cluster)):
            model = self._experts[sort_cluster[i]]

            if output_variances:
                predicted_values.append(model.predict_variances(np.atleast_2d(x[i]))[0])
            else:
                predicted_values.append(model.predict_values(np.atleast_2d(x[i]))[0])

        predicted_values = np.array(predicted_values)

        return predicted_values

    def _predict_smooth_output(self, x, distribs=None, output_variances=False):
        """
        This method predicts the output of x with a smooth recombination.

        Arguments:
        ----------
        - x: np.ndarray
            x samples
        - distribs: distribution list (optional)
            array of membership distributions (use self ones if None)

        Returns
        -------
        - predicted_values : array_like
            predicted output

        """
        predicted_values = []
        if distribs is None:
            distribs = self.distribs
        sort_proba = self._proba_cluster(x, distribs)

        for i in range(len(sort_proba)):
            recombined_value = 0
            for j in range(len(self._experts)):

                if output_variances:
                    expert_value = (
                        self._experts[j].predict_variances(np.atleast_2d(x[i]))[0]
                        * sort_proba[i][j] ** 2
                    )
                else:
                    expert_value = (
                        self._experts[j].predict_values(np.atleast_2d(x[i]))[0]
                        * sort_proba[i][j]
                    )

                recombined_value += expert_value

            predicted_values.append(recombined_value)

        predicted_values = np.array(predicted_values)
        return predicted_values

    @staticmethod
    def _extract_part(values, quantile):
        """
        Divide the values list in quantile parts to return one part
        of (num/quantile) values out of num values.

        Arguments
        ----------
        - values : np.ndarray[num, -1]
            the values list to extract from
        - quantile : int
            the quantile

        Returns
        -------
        - extracted, remaining : np.ndarray, np.ndarray
            the extracted values part, the remaining values

        """
        num = values.shape[0]
        indices = np.arange(0, num, quantile)  # uniformly distributed
        mask = np.zeros(num, dtype=bool)
        mask[indices] = True
        return values[mask], values[~mask]

    def _find_best_model(self, clustered_values):
        """
        Find the best model which minimizes the errors.

        Arguments :
        ------------
        - clustered_values: array_like
            training samples [[X1,X2, ..., Xn, Y], ... ]

        Returns :
        ---------
        - model : surrogate model
            best trained surrogate model

        """
        dim = self.ndim
        clustered_values = np.array(clustered_values)

        scores = {}
        sms = {}

        # validation with 10% of the training data
        test_values, training_values = self._extract_part(clustered_values, 10)

        for name, sm_class in self._enabled_expert_types.items():
            kwargs = {}
            if name in ["RMTB", "RMTC"]:
                # Note: RMTS checks for xlimits,
                # we take limits on all x (not just the trained_values ones) as
                # the surrogate is finally re-trained on the whole x set.
                xlimits = np.zeros((dim, 2))
                for i in range(dim):
                    xlimits[i][0] = np.amin(self.x[:, i])
                    xlimits[i][1] = np.amax(self.x[:, i])
                kwargs = {"xlimits": xlimits}

            sm = sm_class(**kwargs)
            sm.options["print_global"] = False
            sm.set_training_values(training_values[:, 0:dim], training_values[:, dim])
            sm.train()

            expected = test_values[:, dim]
            actual = sm.predict_values(test_values[:, 0:dim])
            l_two = np.linalg.norm(expected - actual, 2)
            # l_two_rel = l_two / np.linalg.norm(expected, 2)
            # mse = (l_two**2) / len(expected)
            # rmse = mse ** 0.5
            scores[sm.name] = l_two
            print(sm.name, l_two)
            sms[sm.name] = sm

        best_name = None
        best_score = None
        for name, rmse in scores.items():
            if best_score is None or rmse < best_score:
                best_name, best_score = name, rmse

        print("Best expert = {}".format(best_name))
        return sms[best_name]

    def _find_best_heaviside_factor(self, x, y):
        """
        Find the best heaviside factor to smooth approximated values.

        Arguments
        ---------
        - x: array_like
            input training samples
        - y: array_like
            output training samples

        Returns
        -------
        hfactor : float
            best heaviside factor wrt given samples

        """
        heaviside_factor = 1.0
        if self.n_clusters > 1:
            hfactors = np.linspace(0.1, 2.1, num=21)
            errors = []
            for hfactor in hfactors:
                distribs = self._create_clusters_distributions(hfactor)
                ypred = self._predict_smooth_output(x, distribs)
                err_rel = np.linalg.norm(y - ypred, 2) / np.linalg.norm(y, 2)
                errors.append(err_rel)
            if max(errors) < 1e-6:
                heaviside_factor = 1.0
            else:
                min_error_index = errors.index(min(errors))
                heaviside_factor = hfactors[min_error_index]
        return heaviside_factor

    """
    Functions related to clustering
    """

    def _create_clusters_distributions(self, heaviside_factor=1.0):
        """
        Create an array of frozen multivariate normal distributions (distribs).

        Arguments
        ---------
        - heaviside_factor: float
            Heaviside factor used to scale covariance matrices

        Returns:
        --------
        - distribs: array_like
            Array of frozen multivariate normal distributions
            with clusters means and covariances

        """
        distribs = []
        dim = self.ndim
        means = self.cluster.means_
        if OLD_SKLEARN:
            cov = heaviside_factor * self.cluster.covars_
        else:
            cov = heaviside_factor * self.cluster.covariances_
        for k in range(self.n_clusters):
            meansk = means[k][0:dim]
            covk = cov[k][0:dim, 0:dim]
            mvn = multivariate_normal(meansk, covk)
            distribs.append(mvn)
        return distribs

    def _cluster_values(self, values, classifier):
        """
        Classify values regarding the given classifier info.

        Arguments
        ---------
        - values: array_like
            values to cluster
        - classifier: array_like
            Cluster corresponding to each point of value in the same order

        Returns
        -------
        - clustered: array_like
            Samples sort by cluster

        Example:
        ---------
        values:
        [[  1.67016597e-01   5.42927264e-01   9.25779645e+00]
        [  5.20618344e-01   9.88223010e-01   1.51596837e+02]
        [  6.09979830e-02   2.66824984e-01   1.17890707e+02]
        [  9.62783472e-01   7.36979149e-01   7.37641826e+01]
        [  3.01194132e-01   8.58084068e-02   4.88696602e+01]
        [  6.40398203e-01   6.91090937e-01   8.91963162e+01]
        [  7.90710374e-01   1.40464471e-01   1.89390766e+01]
        [  4.64498124e-01   3.61009635e-01   1.04779656e+01]]

        cluster_classifier:
        [1 0 0 2 1 2 1 1]

        clustered:
        [[array([   0.52061834,    0.98822301,  151.59683723]),
          array([  6.09979830e-02,   2.66824984e-01,   1.17890707e+02])]
         [array([ 0.1670166 ,  0.54292726,  9.25779645]),
          array([  0.30119413,   0.08580841,  48.86966023]),
          array([  0.79071037,   0.14046447,  18.93907662]),
          array([  0.46449812,   0.36100964,  10.47796563])]
         [array([  0.96278347,   0.73697915,  73.76418261]),
          array([  0.6403982 ,   0.69109094,  89.19631619])]]
        """
        num = len(classifier)
        assert values.shape[0] == num

        clusters = [[] for n in range(self.n_clusters)]
        for i in range(num):
            clusters[classifier[i]].append(values[i])
        return clusters

    def _proba_cluster_one_sample(self, x, distribs):
        """
        Compute membership probabilities to each cluster for one sample.

        Arguments
        ---------
        - x: array_like
            a sample for which probabilities must be calculated
        - distribs: multivariate_normal objects list
            array of normal distributions

        Returns
        -------
        - prob: array_like
            x membership probability for each cluster
        """
        weights = np.array(self.cluster.weights_)
        rvs = np.array([distribs[k].pdf(x) for k in range(len(weights))])

        probs = weights * rvs
        rad = np.sum(probs)

        if rad > 0:
            probs = probs / rad
        return probs

    def _proba_cluster(self, x, distribs=None):
        """
        Calculate membership probabilities to each cluster for each sample
        Arguments
        ---------
        - x: array_like
            samples where probabilities must be calculated

        - distribs : multivariate_normal objects list (optional)
            array of membership distributions. If None, use self ones.

        Returns
        -------
        - probs: array_like
            x membership probabilities to each cluster.

        Examples :
        ----------
        x:
        [[ 0.  0.]
         [ 0.  1.]
         [ 1.  0.]
         [ 1.  1.]]

        prob:
        [[  1.49050563e-02   9.85094944e-01]
         [  9.90381299e-01   9.61870088e-03]
         [  9.99208990e-01   7.91009759e-04]
         [  1.48949963e-03   9.98510500e-01]]
        """

        if distribs is None:
            distribs = self.distribs
        if self.n_clusters == 1:
            probs = np.ones((x.shape[0], 1))
        else:
            probs = np.array(
                [self._proba_cluster_one_sample(x[i], distribs) for i in range(len(x))]
            )

        return probs
