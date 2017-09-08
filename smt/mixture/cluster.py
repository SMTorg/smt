"""
This file contains functions used for the clustering
"""
import warnings
from sklearn import mixture
from sklearn import cluster
from scipy import stats as sct
from scipy.linalg import solve_triangular, cholesky
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_clustering(x, c, n_component=1, method='GMM'):
    """
    Cluster values
    Parameters:
    -----------
    - x: array_like
    Inputs training samples
    - c: array_like
    Input weights
    Optional:
    -----------
    - n_component: int
    Number of clusters. If None, set to 1
    - method: str
    Name of the clustering method can be :
    KMeans,GMM. If None, set to GMM
    """
    values = np.c_[x, c]
    if method is 'KMeans':  # pragma: no cover # NE MARCHE PAS
        clustering = cluster.KMeans(n_component)
        clustering.fit(values)
        return clustering
    if method is 'GMM':
        clustering = mixture.GMM(n_components=n_component,
                                 covariance_type='full', n_init=20)
        clustering.fit(values)
        return clustering


def sort_values_by_cluster(values, number_cluster, sort_cluster):
    """
    Sort values in each cluster
    Parameters
    ---------
    - values: array_like
    Samples to sort
    - number_cluster: int
    Number of cluster
    - sort_cluster: array_like
    Cluster corresponding to each point of value in the same order
    Returns:
    --------
    - sorted_values: array_like
    Samples sort by cluster
    Example:
    ---------
    values:
    [[  1.67016597e-01   5.42927264e-01   9.25779645e+00]
    [  5.20618344e-01   9.88223010e-01   1.51596837e+02]
    [  6.09979830e-02   2.66824984e-01   1.17890707e+02]
    [  9.62783472e-01   7.36979149e-01   7.37641826e+01]
    [  2.65769081e-01   8.09156235e-01   3.43656373e+01]
    [  8.49975570e-01   4.20496285e-01   3.48434265e+01]
    [  3.01194132e-01   8.58084068e-02   4.88696602e+01]
    [  6.40398203e-01   6.91090937e-01   8.91963162e+01]
    [  7.90710374e-01   1.40464471e-01   1.89390766e+01]
    [  4.64498124e-01   3.61009635e-01   1.04779656e+01]]

    number_cluster:
    3

    sort_cluster:
    [1 0 0 2 1 1 1 2 1 1]

    sorted_values
    [[array([   0.52061834,    0.98822301,  151.59683723]),
      array([  6.09979830e-02,   2.66824984e-01,   1.17890707e+02])]
     [array([ 0.1670166 ,  0.54292726,  9.25779645]),
      array([  0.26576908,   0.80915623,  34.36563727]),
      array([  0.84997557,   0.42049629,  34.8434265 ]),
      array([  0.30119413,   0.08580841,  48.86966023]),
      array([  0.79071037,   0.14046447,  18.93907662]),
      array([  0.46449812,   0.36100964,  10.47796563])]
     [array([  0.96278347,   0.73697915,  73.76418261]),
      array([  0.6403982 ,   0.69109094,  89.19631619])]]
    """
    sorted_values = []
    for i in range(number_cluster):
        sorted_values.append([])
    for i in range(len(sort_cluster)):
        sorted_values[sort_cluster[i]].append(values[i].tolist())
    return np.array(sorted_values)


def create_multivar_normal_dis(dim, means, cov):
    """
    Create an array of frozen multivariate normal distributions
    Parameters
    ---------
    - dim: integer
    Dimension of the problem
    - means: array_like
    Array of means
    - cov: array_like
    Array of covariances
    Returns:
    --------
    - gauss_array: array_like
    Array of frozen multivariate normal distributions with means and covariances of the input
    """
    gauss_array = []
    for k in range(len(means)):
        meansk = means[k][0:dim]
        covk = cov[k][0:dim, 0:dim]
        rv = sct.multivariate_normal(meansk, covk, True)
        gauss_array.append(rv)
    return gauss_array


def _proba_cluster_one_sample(weight, gauss_list, x):
    """
    Calculate membership probabilities to each cluster for one sample
    Parameters :
    ------------
    - weight: array_like
    Weight of each cluster
    - gauss_list : multivariate_normal object
    Array of frozen multivariate normal distributions
    - x: array_like
    The point where probabilities must be calculated
    Returns :
    ----------
    - prob: array_like
    Membership probabilities to each cluster for one input
    - clus: int
    Membership to one cluster for one input
    """
    prob = []
    rad = 0

    for k in range(len(weight)):
        rv = gauss_list[k].pdf(
            x)
        val = weight[k] * (rv)
        rad = rad + val
        prob.append(val)

    if rad != 0:
        for k in range(len(weight)):
            prob[k] = prob[k] / rad

    clus = prob.index(max(prob))
    return prob, clus


def _derive_proba_cluster_one_sample(weight, gauss_list, x):
    """
    Calculate the derivation term of the membership probabilities to each cluster for one sample
    Parameters :
    ------------
    - weight: array_like
    Weight of each cluster
    - gauss_list : multivariate_normal object
    Array of frozen multivariate normal distributions
    - x: array_like
    The point where probabilities must be calculated
    Returns :
    ----------
    - derive_prob: array_like
    Derivation term of the membership probabilities to each cluster for one input
    """
    derive_prob = []
    v = 0
    vprime = 0

    for k in range(len(weight)):
        v = v + weight[k] * gauss_list[k].pdf(
            x)
        sigma = gauss_list[k].cov
        invSigma = np.linalg.inv(sigma)
        der = np.dot((x - gauss_list[k].mean), invSigma)
        vprime = vprime - weight[k] * gauss_list[k].pdf(
            x) * der

    for k in range(len(weight)):
        u = weight[k] * gauss_list[k].pdf(
            x)
        sigma = gauss_list[k].cov
        invSigma = np.linalg.inv(sigma)
        der = np.dot((x - gauss_list[k].mean), invSigma)
        uprime = - u * der
        derive_prob.append((v * uprime - u * vprime) / (v**2))

    return derive_prob


def proba_cluster(dim, weight, gauss_list, x):
    """
    Calculate membership probabilities to each cluster for each sample
    Parameters :
    ------------
    - dimension:int
    Dimension of Input samples
    - weight: array_like
    Weight of each cluster
    - gauss_list : multivariate_normal object
    Array of frozen multivariate normal distributions
    - x: array_like
    Samples where probabilities must be calculated
    Returns :
    ----------
    - prob: array_like
    Membership probabilities to each cluster for each sample
    - clus: array_like
    Membership to one cluster for each sample
    Examples :
    ----------
    weight:
    [ 0.60103817  0.39896183]

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

    clus:
    [1 0 0 1]

    """
    n = len(weight)
    prob = []
    clus = []

    for i in range(len(x)):

        if n == 1:
            prob.append([1])
            clus.append(0)

        else:
            proba, cluster = _proba_cluster_one_sample(
                weight, gauss_list, x[i])
            prob.append(proba)
            clus.append(cluster)

    return np.array(prob), np.array(clus)


def derive_proba_cluster(dim, weight, gauss_list, x):
    """
    Calculate the derivation term of the  membership probabilities to each cluster for each sample
    Parameters :
    ------------
    - dim:int
    Dimension of Input samples
    - weight: array_like
    Weight of each cluster
    - gauss_list : multivariate_normal object
    Array of frozen multivariate normal distributions
    - x: array_like
    Samples where probabilities must be calculated
    Returns :
    ----------
    - der_prob: array_like
    Derivation term of the membership probabilities to each cluster for each sample
    """
    n = len(weight)
    der_prob = []

    for i in range(len(x)):

        if n == 1:
            der_prob.append([0])

        else:
            der_proba = _derive_proba_cluster_one_sample(
                weight, gauss_list, x[i])
            der_prob.append(der_proba)

    return np.array(der_prob)
