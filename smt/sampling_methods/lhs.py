"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

LHS sampling; uses the pyDOE3 package.
"""
from pyDOE3 import lhs
from scipy.spatial.distance import pdist, cdist
import numpy as np

from smt.sampling_methods.sampling_method import ScaledSamplingMethod


class LHS(ScaledSamplingMethod):
    def _initialize(self, **kwargs):
        self.options.declare(
            "criterion",
            "c",
            values=[
                "center",
                "maximin",
                "centermaximin",
                "correlation",
                "c",
                "m",
                "cm",
                "corr",
                "ese",
            ],
            types=str,
            desc="criterion used to construct the LHS design "
            + "c, m, cm and corr are abbreviation of center, maximin, centermaximin and correlation, respectively",
        )
        self.options.declare(
            "random_state",
            types=(type(None), int, np.random.RandomState),
            desc="Numpy RandomState object or seed number which controls random draws",
        )

        # Update options values passed by the user here to get 'random_state' option
        self.options.update(kwargs)

        # RandomState is and has to be initialized once at constructor time,
        # not in _compute to avoid yielding the same dataset again and again
        if isinstance(self.options["random_state"], np.random.RandomState):
            self.random_state = self.options["random_state"]
        elif isinstance(self.options["random_state"], int):
            self.random_state = np.random.RandomState(self.options["random_state"])
        else:
            self.random_state = np.random.RandomState()

    def _compute(self, nt):
        """
        Implemented by sampling methods to compute the requested number of sampling points.

        The number of dimensions (nx) is determined based on `xlimits.shape[0]`.

        Arguments
        ---------
        nt : int
            Number of points requested.

        Returns
        -------
        ndarray[nt, nx]
            The sampling locations in the unit hypercube.
        """
        xlimits = self.options["xlimits"]
        nx = xlimits.shape[0]

        if self.options["criterion"] != "ese":
            return lhs(
                nx,
                samples=nt,
                criterion=self.options["criterion"],
                random_state=self.random_state,
            )
        elif self.options["criterion"] == "ese":
            return self._ese(nx, nt)

    def _maximinESE(
        self,
        X,
        T0=None,
        outer_loop=None,
        inner_loop=None,
        J=20,
        tol=1e-3,
        p=10,
        return_hist=False,
        fixed_index=[],
    ):
        """
        Returns an optimized design starting from design X. For more information,
        see R. Jin, W. Chen and A. Sudjianto (2005):
        An efficient algorithm for constructing optimal design of computer
        experiments. Journal of Statistical Planning and Inference, 134:268-287.


        Parameters
        ----------

        X : array
            The design to be optimized

        T0 : double, optional
        Initial temperature of the algorithm.
        If set to None, a standard temperature is used.

        outer_loop : integer, optional
        The number of iterations of the outer loop. If None, set to
        min(1.5*dimension of LHS, 30)

        inner_loop : integer, optional
        The number of iterations of the inner loop. If None, set to
        min(20*dimension of LHS, 100)

        J : integer, optional
        Number of replications of the plan in the inner loop. Default to 20

        tol : double, optional
        Tolerance for modification of Temperature T. Default to 0.001

        p : integer, optional
        Power used in the calculation of the PhiP criterion. Default to 10

        return_hist : boolean, optional
        If set to True, the function returns information about the behaviour of
        temperature, PhiP criterion and probability of acceptance during the
        process of optimization. Default to False


        Returns
        ------

        X_best : array
        The optimized design

        hist : dictionnary
        If return_hist is set to True, returns a dictionnary containing the phiP
        ('PhiP') criterion, the temperature ('T') and the probability of
        acceptance ('proba') during the optimization.

        """

        # Initialize parameters if not defined
        if T0 is None:
            T0 = 0.005 * self._PhiP(X, p=p)
        if inner_loop is None:
            inner_loop = min(20 * X.shape[1], 100)
        if outer_loop is None:
            outer_loop = min(int(1.5 * X.shape[1]), 30)

        T = T0
        X_ = X[:]  # copy of initial plan
        X_best = X_[:]
        d = X.shape[1]
        PhiP_ = self._PhiP(X_best, p=p)
        PhiP_best = PhiP_

        hist_T = list()
        hist_proba = list()
        hist_PhiP = list()
        hist_PhiP.append(PhiP_best)

        # Outer loop
        for z in range(outer_loop):
            PhiP_oldbest = PhiP_best
            n_acpt = 0
            n_imp = 0

            # Inner loop
            for i in range(inner_loop):
                modulo = (i + 1) % d
                l_X = list()
                l_PhiP = list()

                # Build J different plans with a single exchange procedure
                # See description of PhiP_exchange procedure
                for j in range(J):
                    l_X.append(X_.copy())
                    l_PhiP.append(
                        self._PhiP_exchange(
                            l_X[j], k=modulo, PhiP_=PhiP_, p=p, fixed_index=fixed_index
                        )
                    )

                l_PhiP = np.asarray(l_PhiP)
                k = np.argmin(l_PhiP)
                PhiP_try = l_PhiP[k]

                # Threshold of acceptance
                if PhiP_try - PhiP_ <= T * self.random_state.rand(1)[0]:
                    PhiP_ = PhiP_try
                    n_acpt = n_acpt + 1
                    X_ = l_X[k]

                    # Best plan retained
                    if PhiP_ < PhiP_best:
                        X_best = X_
                        PhiP_best = PhiP_
                        n_imp = n_imp + 1

                hist_PhiP.append(PhiP_best)

            p_accpt = float(n_acpt) / inner_loop  # probability of acceptance
            p_imp = float(n_imp) / inner_loop  # probability of improvement

            hist_T.extend(inner_loop * [T])
            hist_proba.extend(inner_loop * [p_accpt])

            if PhiP_best - PhiP_oldbest < tol:
                # flag_imp = 1
                if p_accpt >= 0.1 and p_imp < p_accpt:
                    T = 0.8 * T
                elif p_accpt >= 0.1 and p_imp == p_accpt:
                    pass
                else:
                    T = T / 0.8
            else:
                # flag_imp = 0
                if p_accpt <= 0.1:
                    T = T / 0.7
                else:
                    T = 0.9 * T

        hist = {"PhiP": hist_PhiP, "T": hist_T, "proba": hist_proba}

        if return_hist:
            return X_best, hist
        else:
            return X_best

    def _PhiP(self, X, p=10):
        """
        Calculates the PhiP criterion of the design X with power p.

        X : array_like
        The design where to calculate PhiP
        p : integer
        The power used for the calculation of PhiP (default to 10)
        """

        return ((pdist(X) ** (-p)).sum()) ** (1.0 / p)

    def _PhiP_exchange(self, X, k, PhiP_, p, fixed_index):
        """
        Modifies X with a single exchange algorithm and calculates the corresponding
        PhiP criterion. Internal use.
        Optimized calculation of the PhiP criterion. For more information, see:
        R. Jin, W. Chen and A. Sudjianto (2005):
        An efficient algorithm for constructing optimal design of computer
        experiments. Journal of Statistical Planning and Inference, 134:268-287.

        Parameters
        ----------

        X : array_like
        The initial design (will be modified during procedure)

        k : integer
        The column where the exchange is proceeded

        PhiP_ : double
        The PhiP criterion of the initial design X

        p : integer
        The power used for the calculation of PhiP


        Returns
        ------

        res : double
        The PhiP criterion of the modified design X

        """

        # Choose two (different) random rows to perform the exchange
        i1 = self.random_state.randint(X.shape[0])
        while i1 in fixed_index:
            i1 = self.random_state.randint(X.shape[0])

        i2 = self.random_state.randint(X.shape[0])
        while i2 == i1 or i2 in fixed_index:
            i2 = self.random_state.randint(X.shape[0])

        X_ = np.delete(X, [i1, i2], axis=0)

        dist1 = cdist([X[i1, :]], X_)
        dist2 = cdist([X[i2, :]], X_)
        d1 = np.sqrt(
            dist1**2 + (X[i2, k] - X_[:, k]) ** 2 - (X[i1, k] - X_[:, k]) ** 2
        )
        d2 = np.sqrt(
            dist2**2 - (X[i2, k] - X_[:, k]) ** 2 + (X[i1, k] - X_[:, k]) ** 2
        )

        res = (
            PhiP_**p + (d1 ** (-p) - dist1 ** (-p) + d2 ** (-p) - dist2 ** (-p)).sum()
        ) ** (1.0 / p)
        X[i1, k], X[i2, k] = X[i2, k], X[i1, k]

        return res

    def _ese(self, dim, nt, fixed_index=[], P0=[]):
        """
        Parameters
        ----------

        fixed_index : list
            When running an "ese" optimization, we can fix the indexes of
            the points that we do not want to modify

        """
        # Parameters of maximinESE procedure
        if len(fixed_index) == 0:
            P0 = lhs(dim, nt, criterion=None, random_state=self.random_state)
        else:
            P0 = P0
            self.random_state = np.random.RandomState()
        J = 20
        outer_loop = min(int(1.5 * dim), 30)
        inner_loop = min(20 * dim, 100)

        P, _ = self._maximinESE(
            P0,
            outer_loop=outer_loop,
            inner_loop=inner_loop,
            J=J,
            tol=1e-3,
            p=10,
            return_hist=True,
            fixed_index=fixed_index,
        )
        return P

    def expand_lhs(self, x, n_points, method="basic"):
        """
        Given a Latin Hypercube Sample (LHS) "x", returns an expanded LHS
        by adding "n_points" new points.

        Parameters
        ----------
        x : array
            Initial LHS.
        n_points : integer
            Number of points that are to be added to the expanded LHS.
        method : str, optional
            Methodoly for the construction of the expanded LHS.
            The default is "basic". The other option is "ese" to use the
            ese optimization

        Returns
        -------
        x_new : array
            Expanded LHS.

        """

        xlimits = self.options["xlimits"]

        new_num = len(x) + n_points
        if new_num % len(x) != 0:
            print(
                "WARNING: The added number of points is not a "
                "multiple of the initial number of points."
                "Thus, it cannot be ensured that the output is an LHS."
            )

        # Evenly spaced intervals with the final dimension of the LHS
        intervals = []
        for i in range(len(xlimits)):
            intervals.append(np.linspace(xlimits[i][0], xlimits[i][1], new_num + 1))

        # Creates a subspace with the rows and columns that have no points
        # in the new space
        subspace_limits = [[]] * len(xlimits)
        subspace_bool = []
        for i in range(len(xlimits)):
            subspace_limits[i] = []

            subspace_bool.append(
                [
                    [
                        intervals[i][j] < x[kk][i] < intervals[i][j + 1]
                        for kk in range(len(x))
                    ]
                    for j in range(len(intervals[i]) - 1)
                ]
            )

            [
                subspace_limits[i].append([intervals[i][ii], intervals[i][ii + 1]])
                for ii in range(len(subspace_bool[i]))
                if not (True in subspace_bool[i][ii])
            ]

        # Sampling of the new subspace
        sampling_new = LHS(xlimits=np.array([[0.0, 1.0]] * len(xlimits)))
        x_subspace = sampling_new(n_points)

        column_index = 0
        sorted_arr = x_subspace[x_subspace[:, column_index].argsort()]

        for j in range(len(xlimits)):
            for i in range(len(sorted_arr)):
                sorted_arr[i, j] = subspace_limits[j][i][0] + sorted_arr[i, j] * (
                    subspace_limits[j][i][1] - subspace_limits[j][i][0]
                )

        H = np.zeros_like(sorted_arr)
        for j in range(len(xlimits)):
            order = np.random.permutation(len(sorted_arr))
            H[:, j] = sorted_arr[order, j]

        x_new = np.concatenate((x, H), axis=0)

        if method == "ese":
            # Sampling of the new subspace
            sampling_new = LHS(xlimits=xlimits, criterion="ese")
            x_new = sampling_new._ese(
                len(x_new), len(x_new), fixed_index=np.arange(0, len(x), 1), P0=x_new
            )

        return x_new
