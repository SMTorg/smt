import numpy as np
from math import factorial
from scipy.spatial.distance import pdist,cdist
from scipy.special import binom
import time

__all__ = ['lhs']

def lhs(n, samples=None, criterion=None, iterations=None):
    """
    Generate a latin-hypercube design

    Parameters
    ----------
    n : int
        The number of factors to generate samples for

    Optional
    --------
    samples : int
        The number of samples to generate for each factor (Default: n)
    criterion : str
        Allowable values are "center" or "c", "maximin" or "m",
        "centermaximin" or "cm", and "correlation" or "corr". If no value
        given, the design is simply randomized.
    iterations : int
        The number of iterations in the maximin and correlations algorithms
        (Default: 5).

    Returns
    -------
    H : 2d-array
        An n-by-samples design matrix that has been normalized so factor values
        are uniformly spaced between zero and one.

    Example
    -------
    A 3-factor design (defaults to 3 samples)::

        >>> lhs(3)
        array([[ 0.40069325,  0.08118402,  0.69763298],
               [ 0.19524568,  0.41383587,  0.29947106],
               [ 0.85341601,  0.75460699,  0.360024  ]])

    A 4-factor design with 6 samples::

        >>> lhs(4, samples=6)
        array([[ 0.27226812,  0.02811327,  0.62792445,  0.91988196],
               [ 0.76945538,  0.43501682,  0.01107457,  0.09583358],
               [ 0.45702981,  0.76073773,  0.90245401,  0.18773015],
               [ 0.99342115,  0.85814198,  0.16996665,  0.65069309],
               [ 0.63092013,  0.22148567,  0.33616859,  0.36332478],
               [ 0.05276917,  0.5819198 ,  0.67194243,  0.78703262]])

    A 2-factor design with 5 centered samples::

        >>> lhs(2, samples=5, criterion='center')
        array([[ 0.3,  0.5],
               [ 0.7,  0.9],
               [ 0.1,  0.3],
               [ 0.9,  0.1],
               [ 0.5,  0.7]])

    A 3-factor design with 4 samples where the minimum distance between
    all samples has been maximized::

        >>> lhs(3, samples=4, criterion='maximin')
        array([[ 0.02642564,  0.55576963,  0.50261649],
               [ 0.51606589,  0.88933259,  0.34040838],
               [ 0.98431735,  0.0380364 ,  0.01621717],
               [ 0.40414671,  0.33339132,  0.84845707]])

    A 4-factor design with 5 samples where the samples are as uncorrelated
    as possible (within 10 iterations)::

        >>> lhs(4, samples=5, criterion='correlate', iterations=10)

    """
    H = None

    if samples is None:
        samples = n

    if criterion is not None:
        assert criterion.lower() in ('center', 'c', 'maximin', 'm',
            'centermaximin', 'cm', 'correlation',
            'corr'), 'Invalid value for "criterion": {}'.format(criterion)
    else:
        H = _lhsclassic(n, samples)

    if criterion is None:
        criterion = 'center'

    if iterations is None:
        iterations = 5

    if H is None:
        if criterion.lower() in ('center', 'c'):
            H = _lhscentered(n, samples)
        elif criterion.lower() in ('maximin', 'm'):
            H = _lhsmaximin(n, samples, iterations, 'maximin')
        elif criterion.lower() in ('centermaximin', 'cm'):
            H = _lhsmaximin(n, samples, iterations, 'centermaximin')
        elif criterion.lower() in ('correlate', 'corr'):
            H = _lhscorrelate(n, samples, iterations)

    return H

################################################################################

def _lhsclassic(n, samples):
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    u = np.random.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    rdpoints = np.zeros_like(u)
    for j in range(n):
        rdpoints[:, j] = u[:, j]*(b-a) + a

    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = np.random.permutation(range(samples))
        H[:, j] = rdpoints[order, j]

    return H

################################################################################

def _lhscentered(n, samples):
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    u = np.random.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    _center = (a + b)/2

    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(n):
        H[:, j] = np.random.permutation(_center)

    return H

################################################################################

def _lhsmaximin(n, samples, iterations, lhstype):
    maxdist = 0

    # Maximize the minimum distance between points
    for i in range(iterations):
        if lhstype=='maximin':
            Hcandidate = _lhsclassic(n, samples)
        else:
            Hcandidate = _lhscentered(n, samples)

        d = pdist(Hcandidate)
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = Hcandidate.copy()

    return H

################################################################################

def _lhscorrelate(n, samples, iterations):
    mincorr = np.inf

    # Minimize the components correlation coefficients
    for i in range(iterations):
        # Generate a random LHS
        Hcandidate = _lhsclassic(n, samples)
        R = np.corrcoef(Hcandidate)
        if np.max(np.abs(R[R!=1]))<mincorr:
            mincorr = np.max(np.abs(R-np.eye(R.shape[0])))
            print 'new candidate solution found with max,abs corrcoef = {}'.format(mincorr)
            H = Hcandidate.copy()

    return H

################################################################################

def lhs_maximinESE(X, T0=None, outer_loop=None, inner_loop=None, J=20,
               tol=1e-3, p=10, return_hist=False, fixed_index=[]):
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
        T0 = 0.005*PhiP(X, p=p)
    if inner_loop is None:
        inner_loop = min(20*X.shape[1], 100)
    if outer_loop is None:
        outer_loop = min(int(1.5*X.shape[1]), 30)

    T = T0
    X_ = X[:] # copy of initial plan
    X_best = X_[:]
    d = X.shape[1]
    PhiP_ = PhiP(X_best, p=p)
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

            modulo = (i+1)%d
            l_X = list()
            l_PhiP = list()

            # Build J different plans with a single exchange procedure
            # See description of PhiP_exchange procedure
            for j in range(J):
                l_X.append(X_.copy())
                l_PhiP.append(_PhiP_exchange(l_X[j], k=modulo, PhiP_=PhiP_, p=p,
                                             fixed_index=fixed_index))

            l_PhiP = np.asarray(l_PhiP)
            k = np.argmin(l_PhiP)
            PhiP_try = l_PhiP[k]

            # Threshold of acceptance
            if PhiP_try - PhiP_ <= T * np.random.rand(1)[0]:
                PhiP_ = PhiP_try
                n_acpt = n_acpt + 1
                X_ = l_X[k]

                # Best plan retained
                if PhiP_ < PhiP_best:
                    X_best = X_
                    PhiP_best = PhiP_
                    n_imp = n_imp + 1

            hist_PhiP.append(PhiP_best)


        p_accpt = float(n_acpt) / inner_loop # probability of acceptance
        p_imp = float(n_imp) / inner_loop # probability of improvement

        hist_T.extend(inner_loop*[T])
        hist_proba.extend(inner_loop*[p_accpt])

        if PhiP_best - PhiP_oldbest < tol:
            # flag_imp = 1
            if p_accpt>=0.1 and p_imp<p_accpt:
                T = 0.8*T
            elif p_accpt>=0.1 and p_imp==p_accpt:
                pass
            else:
                T = T/0.8
        else:
            # flag_imp = 0
            if p_accpt<=0.1:
                T = T/0.7
            else:
                T = 0.9*T

    hist = {'PhiP': hist_PhiP, 'T': hist_T, 'proba': hist_proba}

    if return_hist:
        return X_best, hist
    else:
        return X_best



def PhiP(X, p=10):
    """
    Calculates the PhiP criterion of the design X with power p.

    X : array_like
        The design where to calculate PhiP
    p : integer
        The power used for the calculation of PhiP (default to 10)
    """

    return ((pdist(X)**(-p)).sum()) ** (1./p)


def _PhiP_exchange(X, k, PhiP_, p, fixed_index):
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
    i1 = np.random.randint(X.shape[0])
    while i1 in fixed_index:
        i1 = np.random.randint(X.shape[0])

    i2 = np.random.randint(X.shape[0])
    while i2 == i1 or i2 in fixed_index:
        i2 = np.random.randint(X.shape[0])

    X_ = np.delete(X, [i1,i2], axis=0)

    dist1 = cdist([X[i1,:]], X_)
    dist2 = cdist([X[i2,:]], X_)
    d1 = np.sqrt(dist1**2 + (X[i2,k] - X_[:,k])**2 - (X[i1,k] - X_[:,k])**2)
    d2 = np.sqrt(dist2**2 - (X[i2,k] - X_[:,k])**2 + (X[i1,k] - X_[:,k])**2)

    res = (PhiP_**p + (d1**(-p) - dist1**(-p) + d2**(-p) - dist2**(-p)).sum())**(1./p)
    X[i1,k], X[i2,k] = X[i2,k], X[i1,k]

    return res

def doe_remi(dim,n):
    # Parameters of maximinESE procedure
    P0 = lhs(dim, n, criterion = None)
    J = 20
    outer_loop = min(int(1.5*dim), 30)
    inner_loop = min(20*dim, 100)



    D0 = pdist(P0)
    R0 = np.corrcoef(P0)
    corr0 = np.max(np.abs(R0[R0!=1]))
    phip0 = PhiP(P0)

    t1 = time.time()
    print("\nCalculation of 'maximinESE' design...")
    P, historic = lhs_maximinESE(P0, outer_loop=outer_loop, inner_loop=inner_loop,
                                 J=J, tol=1e-3, p=10, return_hist=True)
    print('Time: ' + str(time.time()-t1))
    return P


def fullfact(levels):
    """
    Create a general full-factorial design

    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.

    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels 0 to k-1 for a k-level factor

    Example
    -------
    ::

        >>> fullfact([2, 4, 3])
        array([[ 0.,  0.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 1.,  1.,  0.],
               [ 0.,  2.,  0.],
               [ 1.,  2.,  0.],
               [ 0.,  3.,  0.],
               [ 1.,  3.,  0.],
               [ 0.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 0.,  2.,  1.],
               [ 1.,  2.,  1.],
               [ 0.,  3.,  1.],
               [ 1.,  3.,  1.],
               [ 0.,  0.,  2.],
               [ 1.,  0.,  2.],
               [ 0.,  1.,  2.],
               [ 1.,  1.,  2.],
               [ 0.,  2.,  2.],
               [ 1.,  2.,  2.],
               [ 0.,  3.,  2.],
               [ 1.,  3.,  2.]])

    """
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))

    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat /= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j]*level_repeat
        rng = lvl*range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng

    return H
