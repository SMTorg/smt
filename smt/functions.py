from __future__ import division
import numpy as np
from  KPLS_tools.pls import pls as _pls

'''
todo
Check all
'''

def standard(X,y):
    X_mean = np.mean(X, axis=0)
    X_std = X.std(axis=0,ddof=1)
    y_mean = np.mean(y, axis=0)
    y_std = y.std(axis=0,ddof=1)
    X_std[X_std == 0.] = 1.
    y_std[y_std == 0.] = 1.

    # center and scale X
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_std

    return X, y, X_mean, y_mean, X_std, y_std

def l1_cross_distances(X):

    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.

    Parameters
    ----------

    X: array_like
    An array with shape (n_samples, n_features)

    Returns
    -------

    D: array with shape (n_samples * (n_samples - 1) / 2, n_features)
    The array of componentwise L1 cross-distances.

    ij: arrays with shape (n_samples * (n_samples - 1) / 2, 2)
    The indices i and j of the vectors in X associated to the cross-
    distances in D: D[k] = np.abs(X[ij[k, 0]] - Y[ij[k, 1]]).
    """
    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0

    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):n_samples])

    return D, ij.astype(np.int)


def componentwise_distance(D,corr,ncomp,dim,coeff_pls,limit):
    D_final = np.zeros((D.shape[0],ncomp))
    i,nb_limit  = 0,int(limit)

    if ncomp == dim:
        # Kriging
        while True:
            if i * nb_limit > D_final.shape[0]:
                return D_final
            else:
                if corr == 'squar_exp':
                    D_final[i*nb_limit:(i+1)*nb_limit,:] = D[i*nb_limit:(i+1)*
                                                             nb_limit,:]**2
                else:
                    D_final[i*nb_limit:(i+1)*nb_limit,:] = np.abs(D[i*nb_limit:
                                                            (i+1)*nb_limit,:])
                i+=1
    else:
        # KPLS or GEKPLS
        while True:
            if i * nb_limit > D_final.shape[0]:
                return D_final
            else:
                if corr == 'squar_exp':
                    D_final[i*nb_limit:(i+1)*nb_limit,:] = np.dot(D[i*nb_limit:
                                    (i+1)*nb_limit,:]** 2,coeff_pls**2)
                else:
                    D_final[i*nb_limit:(i+1)*nb_limit,:] = np.dot(np.abs(D[i*
                        nb_limit:(i+1)*nb_limit,:]),np.abs(coeff_pls))
                i+=1


def compute_pls(kpls,X,y,opt=0):
    XX = np.empty(shape = (0,kpls.dim))
    yy = np.empty(shape = (0,1))
    pls = _pls(kpls.sm_options['ncomp'])
    if opt == 0:
        #KPLS
        pls.fit(X,y)
        return np.abs(pls.x_rotations_), XX, yy

    elif opt == 1:
        #GEKPLS-indGEKPLS-GEHKPLS
        coeff_pls = np.zeros((kpls.nt,kpls.dim,kpls.sm_options['ncomp']))

        for i in range(kpls.nt):
            _X = np.zeros((kpls.dim+1,kpls.dim))
            _y = np.zeros((kpls.dim+1,1))
            _X[0,:] = X[i,:].copy()
            _y[0,0] = y[i,0].copy()
            for j in range(1,kpls.dim+1):
                _X[j,:] = _X[0,:]
                _X[j,j-1] +=kpls.sm_options['delta_x']*(kpls.sm_options['xlimits'][j-1,1]-
                                                        kpls.sm_options['xlimits'][j-1,0])
                _y[j,0] = _y[0,0].copy()+ kpls.training_pts['exact'][j][1][i,0]* \
                          kpls.sm_options['delta_x']*(kpls.sm_options['xlimits'][j-1,1]-
                                                     kpls.sm_options['xlimits'][j-1,0])
            pls.fit(_X.copy(),_y.copy())

            coeff_pls[i,:,:] = pls.x_rotations_
            if kpls.sm_options['indGEKPLS'] != 0:
                max_coeff = np.argsort(np.abs(coeff_pls[i,:,0]))[-kpls.sm_options['indGEKPLS']:]
                XX = np.vstack((XX,_X[1+max_coeff,:]))
                yy = np.vstack((yy,_y[1+max_coeff,:]))

        return np.abs(coeff_pls).mean(axis=0), XX, yy

# -----------------------------------------------------------------------------
# Argument for the logistic function (quadratic)
# -----------------------------------------------------------------------------

def computeaquadratic(pi, mu, Sigma, x):
    # the quadratic term
    qterm = np.linalg.solve(Sigma, x)
    quadTerm = -1.0/2.0 * np.dot(x, qterm)

    # vector b: for the linear term
    b = np.linalg.solve(Sigma, mu)
    linTerm = np.dot(b,x)

    # vector c: for the constant term
    cterm1 = -1.0/2.0*np.dot(mu, np.linalg.solve(Sigma, mu))
    cterm2 = -1.0/2.0*np.log(np.linalg.det(Sigma))
    cterm3 =  np.log(pi)

    constTerm = cterm1 + cterm2 + cterm3

    a = quadTerm + linTerm + constTerm

    return a


# -----------------------------------------------------------------------------
# Normal PDF
# -----------------------------------------------------------------------------

def normalPDF(x, mu, s2):
    pdf = np.sqrt(1.0/2*np.pi)*np.sqrt(1.0/s2)*np.exp(-((x-mu)**2)/(2*s2))

    return pdf


# -----------------------------------------------------------------------------
# Log likelihood for the mixture of Gaussians (MOG)
# -----------------------------------------------------------------------------

def mog_logLikelihood(x, piVec, muVec, sigma2Vec):
    # number of data
    N = len(x)
    nK = len(piVec)

    # log likelihood
    L = 0.0

    for i in range(N):
        # compute the likelihood of the i-th data
        xi = x[i,0]
        tmp = 0.0
        for k in range(nK):
            pik = piVec[k]
            muk = muVec[k]
            s2k = sigma2Vec[k]

            pdf = normalPDF(xi, muk, s2k)
            tmp += pik * pdf

        if tmp < 1e-5:
            tmp = 1e-5
        L += np.log(tmp)

    return L

# ------------------------------------------------------------------------------
# UNSUPERVISED LEARNING ALGORITHM
# To classify the training samples based on the function values/gradients
# No classification data in the training data
# Training data: {xn}, n = 1, ..., N
# ------------------------------------------------------------------------------

def mixtureofgaussians(nclusters, tol, data):
    # initialize the mean values and variance

    # split the training data into nclusters
    N = len(data)
    ntmp = np.int(N/nclusters)  # number of data in each temporary cluster

    muVec = np.zeros(nclusters)
    sigma2Vec = np.zeros(nclusters)
    piVec = np.zeros(nclusters)
    for i in range(nclusters):
        if (i != nclusters-1):
            xclusters = data[(i*ntmp):((i+1)*ntmp),0]
        else:
            xclusters = data[(i*ntmp):,0]

        muVec[i] = np.mean(xclusters)
        sigma2Vec[i] = (np.std(xclusters))**2

        piVec[i] = 1.0/nclusters

    L = mog_logLikelihood(data, piVec, muVec, sigma2Vec)
    # posterior probabilities p(zk=1|x) = gamma(zk)
    gamma = np.zeros((N, nclusters))

    error = 1.0
    iter = 0
    while (error > tol):
        # E-step, evaluate responsibilities
        for n in range(N):
            xn = data[n,0]
            num = np.zeros(nclusters)
            for k in range(nclusters):
                pik = piVec[k]
                muk = muVec[k]
                s2k = sigma2Vec[k]

                # numerator = p(zk=1)*p(x|zk=1)
                num[k] = pik*normalPDF(xn, muk, s2k)

            # denominator = sum(numerator)
            denom = sum(num)

            # regularization, so the denominator doesn't get too close to zero
            if denom < 1e-5:
                denom = 1e-5

            # normalize
            for k in range(nclusters):
                gamma[n,k] = num[k]/denom


        # M-step, update parameters

        # Nk = sum(gamma[:,k]), the effective number of points assigned to the k-th cluster
        Nk = np.zeros(nclusters)
        for k in range(nclusters):
            # update muk
            Nk[k] = sum(gamma[:,k])

            muk = 0.0
            for n in range(N):
                muk += gamma[n,k]*data[n,0]

            muk = muk/Nk[k]

            muVec[k] = muk

            # update sigma2k
            s2k = 0.0
            for n in range(N):
                s2k += gamma[n,k]*((data[n,0]-muVec[k])**2)

            s2k = s2k/Nk[k]

            sigma2Vec[k] = s2k

            # update pik
            pik = Nk[k]/N
            piVec[k] = pik

        # update the likelihood function
        newL = mog_logLikelihood(data, piVec, muVec, sigma2Vec)
        error = np.abs(newL-L)
        L = newL
        iter += 1
        #print iter, newL

    # list of indices of data that belong to each cluster
    indClusters = []
    for k in range(nclusters):
        indClusters.append([])

    for n in range(N):
        # find the cluster index where the posterior probability is the highest
        indMax = np.argsort(gamma[n,:])
        indMax = indMax[-1]

        indClusters[indMax].append(n)

    return indClusters

# ------------------------------------------------------------------------------
# SUPERVISED LEARNING ALGORITHM
# To provide a classification in the x-space
# Use the classified training samples (from the unsupervised learning algorithm
# results) as the training data.
# Training data: {xn, tn}, n = 1, ..., N
# ------------------------------------------------------------------------------

def gaussianclassifier(x,t,regularize=True, regConstant=0.01):
    # x: data points
    # t: classification inputs

    # problem dimension (number of variables)
    Ndv = x.shape[1]

    # number of data
    N = len(t)

    # number of clusters
    nclusters = len(np.unique(t))

    # number of data within each cluster
    nmembers = np.zeros(nclusters, dtype=int)

    # clusters' prior probabilities
    pi = np.zeros(nclusters)

    # indices of data for each cluster
    indmembers = []
    xmembers = []
    # initialize
    for j in range(nclusters):
        indmembers.append([])
        xmembers.append([])

    for j in range(nclusters):
        indmembers[j] = [i for (i,val) in enumerate(t) if val == j]
        xmembers[j] = x[indmembers[j],:].copy()
        nmembers[j] = len(indmembers[j])

        pi[j] = np.float(nmembers[j])/np.float(N)

    # compute mean of each cluster
    mu = np.zeros((nclusters, Ndv))

    for j in range(nclusters):
        for d in range(Ndv):
            mu[j,d] = np.sum(xmembers[j][:,d])
        mu[j,:] = mu[j,:]/nmembers[j]

    # compute the class covariance matrix
    Sigma = []
    # initialize
    for j in range(nclusters):
        Sigma.append([])

    for j in range(nclusters):
        Sigma[j] = np.zeros((Ndv, Ndv))

        for n in range(nmembers[j]):
            tmp_Sigma = np.zeros((Ndv, Ndv))

            vec = xmembers[j][n,:] - mu[j,:]
            for p in range(Ndv):
                for r in range(Ndv):
                    tmp_Sigma[p,r] = vec[p]*vec[r]

            Sigma[j] += tmp_Sigma

        Sigma[j] = 1.0/np.float(nmembers[j]) * Sigma[j]

    if regularize:
        reg = np.zeros((Ndv, Ndv))

        for d in range(Ndv):
            reg[d,d] = regConstant

        for j in range(nclusters):
            Sigma[j] += reg

    return pi, mu, Sigma

# -----------------------------------------------------------------------------
# Evaluate the posteriors of the Gaussian classifier
# When we already have the pi, mu, and Sigma
# -----------------------------------------------------------------------------

def eval_gaussianClassifier(xevals, pi, mu, Sigma, weight=1.0, bias=0.0):
    Nevals = xevals.shape[0]
    Ndv = xevals.shape[1]
    nclusters = len(pi)

    posteriors = np.zeros((Nevals, nclusters))
    classList = np.zeros(Nevals, dtype=int)
    clusterCount = np.zeros(nclusters, dtype=int)

    for n in range(Nevals):
        x = xevals[n,:]

        post = np.zeros(nclusters)
        for j in range(nclusters):
            # compute the argument of the logistic function
            post[j] = computeaquadratic(pi[j], mu[j,:], Sigma[j], x)
            post[j] = np.exp(weight*post[j]+bias)

        # normalize the posterior probability
        post = post/np.sum(post)

        posteriors[n,:] = post

        # sort the posterior probability from lowest to highest
        indSort = np.argsort(post)

        # find the cluster index with the highest posterior probability
        indClass = indSort[-1]

        classList[n] = indClass

        # add the counter
        clusterCount[indClass] += 1

    indevalsClusters = []
    # initialize
    for j in range(nclusters):
        indevalsClusters.append(np.zeros(clusterCount[j], dtype=int))

    for j in range(nclusters):
        indevalsClusters[j] = [i for (i,val) in enumerate(classList) if val == j]

    return indevalsClusters, posteriors, classList, clusterCount
