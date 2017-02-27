# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 15:17:23 2015

@author: MohamedAmine
"""

# -*- coding: utf-8 -*-
################################################################################
#                               TOOLS                                          #
################################################################################
# \brief                outils doe initial
#
# \author Mohamed Amine Bouhlel
# Bouhlel Mohamed Amine <mohamed.bouhlel@snecma.fr>
#                       <mohamed.amine.bouhlel@gmail.com>
"""
debug

"""

import numpy as np

########################### LHS
def _pdist(x):
    """
calculate the pair-wise point distances of a matrix

    Parameters
    ----------
    x : 2d-array
        An m-by-n array of scalars, where there are m points in n dimensions.

    Returns
    -------
    d : array
        A 1-by-b array of scalars, where b = m*(m - 1)/2. This array contains
        all the pair-wise point distances, arranged in the order (1, 0),
        (2, 0), ..., (m-1, 0), (2, 1), ..., (m-1, 1), ..., (m-1, m-2).

    Examples
    --------
    ::

        >>> x = np.array([[0.1629447, 0.8616334],
        ...               [0.5811584, 0.3826752],
        ...               [0.2270954, 0.4442068],
        ...               [0.7670017, 0.7264718],
        ...               [0.8253975, 0.1937736]])
        >>> pdist(x)
        array([0.6358488, 0.4223272, 0.6189940, 0.9406808, 0.3593699,
              [0.3908118, 0.3087661, 0.6092392, 0.6486001, 0.5358894]])

    """

    x = np.atleast_2d(x)
    assert len(x.shape)==2, 'Input array must be 2d-dimensional'

    m, n = x.shape
    if m<2:
        return []

    d = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            d.append((sum((x[j, :] - x[i, :])**2))**0.5)

    return np.array(d)

def _lhsclassic(n, samples):
    """
generate the intervals
    """
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

def _lhscentered(n, samples):
    """
generate the intervals
    """
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

def _lhsmaximin(n, samples, iterations, lhstype):
    """
maximin
    """
    maxdist = 0
    # Maximize the minimum distance between points
    for i in range(iterations):
        if lhstype=='maximin':
            Hcandidate = _lhsclassic(n, samples)
        else:
            Hcandidate = _lhscentered(n, samples)

        d = _pdist(Hcandidate)
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = Hcandidate.copy()

    return H

def _lhscorrelate(n, samples, iterations):
    """
correlation des points
    """
    mincorr = np.inf

    # Minimize the components correlation coefficients
    for i in range(iterations):
        # Generate a random LHS
        Hcandidate = _lhsclassic(n, samples)
        R = np.corrcoef(Hcandidate)
        if np.max(np.abs(R[R!=1]))<mincorr:
            mincorr = np.max(np.abs(R-np.eye(R.shape[0])))
            #print 'new candidate solution found with max,abs corrcoef = {}'.format(mincorr)
            H = Hcandidate.copy()

    return H

##########################

def trans(X, binf, bsup, opt=1):
    """
transformer [0.1]^dim en [binf, bsup]^dim
    Parameters.
    ----------

    X 2d-array
    A 2d-array (n_samples,n_features) containing the training points.

    binf: list of float
    List of the lower bound of variables.
    bsup: list of float
    List of the upper bound of variables.

    opt int (1 or 2)
    1 to transform [0.1]^dim towards [binf, bsup]^dim.
    2 to transform [binf, bsup]^dim towards [0.1]^dim.

    Returns
    -------
    Y 2d-array
    the transforming X-Array
    """
    A=1./(np.array(bsup)-np.array(binf))
    B=-np.array(binf)*1./(np.array(bsup)-np.array(binf))
    if opt==1:
        Y=(X-B)/A
    elif opt==2:
        Y=A*X+B

    return Y

def lhs(n, samples=None, criterion=None, iterations=None):
    """
    pour plus de details voir:
    https://pypi
generate a latin-hypercube design



    Parameters
    ----------
    n : int
        the number of factors to generate samples for

    Optional
    --------
    samples : int
        the number of samples to generate for each factor (Default: n)
    criterion : str
        Allowable values are "center" or "c", "maximin" or "m",
        "centermaximin" or "cm", and "correlation" or "corr". If no value
        given, the design is simply randomized.
    iterations : int
        the number of iterations in the maximin and correlations algorithms
        (Default: 5).

    Returns
    -------
    H : 2d-array
        An n-by-samples design matrix that has been normalized so factor values
        are uniformly spaced between zero and one.

    Example
    -------
    A 3-factor design (defaults to 3 samples)::

        >>> lhs_(3)
        array([[ 0.40069325,  0.08118402,  0.69763298],
               [ 0.19524568,  0.41383587,  0.29947106],
               [ 0.85341601,  0.75460699,  0.360024  ]])

    A 4-factor design with 6 samples::

        >>> lhs_(4, samples=6)
        array([[ 0.27226812,  0.02811327,  0.62792445,  0.91988196],
               [ 0.76945538,  0.43501682,  0.01107457,  0.09583358],
               [ 0.45702981,  0.76073773,  0.90245401,  0.18773015],
               [ 0.99342115,  0.85814198,  0.16996665,  0.65069309],
               [ 0.63092013,  0.22148567,  0.33616859,  0.36332478],
               [ 0.05276917,  0.5819198 ,  0.67194243,  0.78703262]])

    A 2-factor design with 5 centered samples::

        >>> lhs_(2, samples=5, criterion='center')
        array([[ 0.3,  0.5],
               [ 0.7,  0.9],
               [ 0.1,  0.3],
               [ 0.9,  0.1],
               [ 0.5,  0.7]])

    A 3-factor design with 4 samples where the minimum distance between
    all samples has been maximized::

        >>> lhs_(3, samples=4, criterion='maximin')
        array([[ 0.02642564,  0.55576963,  0.50261649],
               [ 0.51606589,  0.88933259,  0.34040838],
               [ 0.98431735,  0.0380364 ,  0.01621717],
               [ 0.40414671,  0.33339132,  0.84845707]])

    A 4-factor design with 5 samples where the samples are as uncorrelated
    as possible (within 10 iterations)::

        >>> lhs_(4, samples=5, criterion='correlate', iterations=10)

    """
    H = None

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
