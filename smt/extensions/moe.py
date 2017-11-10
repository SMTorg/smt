"""
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.

Mixture of Experts
"""

from __future__ import division
import six
import numpy as np
import warnings
from sklearn import mixture, cluster
from scipy import stats as sct

from smt.utils.options_dictionary import OptionsDictionary
from smt.extensions.extensions import Extensions
from smt.utils.misc import compute_rms_error

warnings.filterwarnings("ignore", category=DeprecationWarning)

class MOE(Extensions):
    
    def _initialize(self):
        super(MOE, self)._initialize()
        declare = self.options.declare

        declare('sms', )
        
        declare('xt', None, types=np.ndarray, desc='Training inputs')
        declare('yt', None, types=np.ndarray, desc='Training outputs')
        declare('c', None, types=np.ndarray, desc='Clustering training outputs')

        declare('xtest', None, types=np.ndarray, desc='Test inputs')
        declare('ytest', None, types=np.ndarray, desc='Test outputs')

        declare('n_clusters', 2, types=int, desc='Number of cluster')
        declare('smooth_recombination', True, types=bool, desc='Continuous cluster transition')
        declare('heaviside_optimization', False, types=bool, 
                desc='Optimize Heaviside scaling factor in cas eof smooth recombination')
        declare('derivatives_support', False, types=bool, 
                desc='Use only experts that support derivatives prediction')        
        declare('variances_support', False, types=bool, 
                desc='Use only experts that support variance prediction')

        for name, smclass in self._surrogate_type.iteritems():
            sm_options = smclass().options
            declare(name+'_options', sm_options._dict, types=dict, desc=name+' options dictionary')

        self.x = None
        self.y = None
        self.c = None

        self.n_clusters = None
        self.smooth_recombination = None
        self.heaviside_optimization = None
        self.heaviside_factor = 1.

        self.experts = []

    def train(self):
        """
        Supports SM api
        """
        super(MOE, self).apply_method()

    def predict_values(self, x):
        """
        Supports SM api
        """
        return super(MOE, self).analyse_results(x=x, operation='predict_values')        

    def _apply(self):
        """
        Build and train the mixture of experts
        """
        self.x = x = self.options['xt']
        self.y = y = self.options['yt']
        self.c = c = self.options['c']
        if not self.c:
            self.c = c = y

        self.n_clusters = self.options['n_clusters']
        self.smooth_recombination = self.options['smooth_recombination']
        self.heaviside_optimization = self.options['smooth_recombination'] and self.options['heaviside_optimization']
        self.heaviside_factor = 1.

        self._check_inputs()

        self.expert_types = self._select_expert_types()
        self.experts = []

        # Set test values and trained values
        xtest = self.options['xtest']
        ytest = self.options['ytest']
        values = np.c_[x, y, c]
        test_data_present = xtest is not None and ytest is not None
        if test_data_present:
            self.test_values = np.c_[xtest, ytest] 
            self.training_values = values
        else:
            self.test_values, self.training_values = self._extract_part(values, 10)

        self.ndim = nx = x.shape[1]
        xt = self.training_values[:, 0:nx]
        yt = self.training_values[:, nx:nx+1]
        ct = self.training_values[:, nx+1:]

        # Clustering
        self.cluster = mixture.GMM(n_components=self.n_clusters,
                                   covariance_type='full', n_init=20)
        self.cluster.fit(np.c_[xt, ct])        
        if not self.cluster.converged_:
            raise Exception('Clustering not converged')

        # Choice of the experts and training
        self._fit(xt, yt, ct)

        xtest = self.test_values[:, 0:nx]
        ytest = self.test_values[:, nx:nx+1] 
        # Heaviside factor
        if self.heaviside_optimization and self.n_clusters > 1:
            self.heaviside_factor = self._find_best_heaviside_factor(xtest, ytest)
            print('BEST HEAVISIDE =', self.heaviside_factor)
            self.gauss = self._create_multivariate_normal(self.heaviside_factor)

        # if nx == 1:
        #     Maxi = max(values[:, 0])
        #     Mini = min(values[:, 0])
        # if nx == 2:
        #     Maxi = np.zeros(2)
        #     Mini = np.zeros(2)
        #     Maxi[0] = max(values[:, 0])
        #     Maxi[1] = max(values[:, 1])
        #     Mini[0] = min(values[:, 0])
        #     Mini[1] = min(values[:, 1])

        # self.plotClusterGMM(Maxi, Mini, x, y, heaviside=self.heaviside_factor)

        self.compute_error(xtest, ytest)

        if not test_data_present:
            # if we have used part of data to validate
            self._fit(x, y, c, new_model=False)


    def _analyse_results(self, x, operation='predict_values', kx=None):
        if operation == 'predict_values':
            if self.smooth_recombination:
                y = self._predict_smooth_output(x)
            else:
                y = self._predict_hard_output(x)
            return y 
        else:
            raise ValueError("MOE supports predict_values operation only.")
        return y
    
    def compute_error(self, x, y):
        """
        Valid the Moe with the input samples
        Parameters:
        -----------
        - x: array_like
        Input testing samples
        - y : array_like
        Output testing samples
        """
        ys = self._predict_smooth_output(x)
        yh = self._predict_hard_output(x)
        self.valid_hard = Error(y, yh)
        self.valid_smooth = Error(y, ys)
    
    @staticmethod
    def _rmse(expected, actual):
        l_two = np.linalg.norm(expected - actual)
        l_two_rel = l_two / np.linalg.norm(expected)
        mse = (l_two**2) / len(expected)
        rmse = mse ** 0.5
        return rmse

    def _check_inputs(self):
        """
        Check the input data given by the client is correct.
        raise Value error with relevant message
        """
        if self.x is None or self.y is None:
            raise ValueError("check x and y values")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("The number of input points %d doesn t match with the number of output points %d."
                             % (self.x.shape[0], self.y.shape[0]))
        if self.y.shape[0] != self.c.shape[0]:
            raise ValueError("The number of output points %d doesn t match with the number of criterion weights %d."
                             % (self.y.shape[0], self.c.shape[0]))
        # choice of number of cluster
        max_n_clusters = int(len(self.x) / 10) + 1
        if self.n_clusters > max_n_clusters:
            print 'Number of clusters should be inferior to {0}'.format(max_n_clusters)
            raise ValueError(
                'The number of clusters is too high considering the number of points')

    def _select_expert_types(self):
        """
        Select relevant surrogate models (experts) regarding MOE options
        """
        prototypes = {name: smclass() for name, smclass in self._surrogate_type.iteritems()}
        if self.options['derivatives_support']:
            prototypes = {name: proto for name, proto in prototypes.iteritems() if proto.support['derivatives']}
        if self.options['variances_support']:
            prototypes = {name: proto for name, proto in prototypes.iteritems() if proto.support['variances']}
        return {name: self._surrogate_type[name] for name in prototypes}

    def _fit(self, x_trained, y_trained, c_trained, new_model=True):
        """
        Find the best model for each cluster (clustering already done) and train it if new_model is True
        Else train the points given (choice of best models by cluster already done)
        Parameters:
        -----------
        - x_trained: array_like
        Input training samples
        - y_trained: array_like
        Output training samples
        - c_trained: array_like
        Clustering training samples
        Optional:
        -----------
        - detail: Boolean
        Set True to see detail of the search
        - new_model : bool
        Set true to search the best local model
        """
        self.gauss = self._create_multivariate_normal(self.heaviside_factor)

        cluster_classifier = self.cluster.predict(np.c_[x_trained, c_trained])

        # sort trained_values for each cluster
        clusters = self._cluster_values(np.c_[x_trained, y_trained], cluster_classifier)

        # find model for each cluster
        for i in range(self.n_clusters):

            if new_model:
                if len(self._surrogate_type) == 1:
                    pass
                    # #
                    # x_trained = np.array(trained_cluster[clus])[:, 0:self.ndim]
                    # y_trained = np.array(trained_cluster[clus])[:, self.ndim]
                    # model = self.use_model()
                    # model.set_training_values(x_trained, y_trained)
                    # model.train()
                else:
                    model = self._find_best_model(clusters[i])

                self.experts.append(model)

            else:  # Train on the overall domain
                trained_values = np.array(clusters[i])
                x_trained = trained_values[:, 0:self.ndim]
                y_trained = trained_values[:, self.ndim]
                self.experts[i].set_training_values(x_trained, y_trained)
                self.experts[i].train()
        
    def _predict_hard_output(self, x):
        """
        This method predicts the output of a x samples for a hard recombination
        Parameters:
        ----------
        - x: Array_like
        x samples
        Return :
        ----------
        - predicted_values : array_like
        predicted output
        """
        predicted_values = []
        _, sort_cluster = self._proba_cluster(
            self.ndim, self.cluster.weights_, self.gauss, x)

        for i in range(len(sort_cluster)):
            model = self.experts[sort_cluster[i]]
            predicted_values.append(model.predict_values(np.atleast_2d(x[i]))[0])
        predicted_values = np.array(predicted_values)

        return predicted_values

    def _predict_smooth_output(self, x, gauss=None):
        """
        This method predicts the output of x with a smooth recombination
        Parameters:
        ----------
        - x: np.ndarray
        x samples
        - gauss: 
        array of frozen multivariate normal distributions (see)
        Returns 
        -------
        - predicted_values : array_like
        predicted output
        """
        predicted_values = []
        if gauss is None:
            gauss = self.gauss
        sort_proba, _ = self._proba_cluster(self.ndim, self.cluster.weights_, gauss, x)

        for i in range(len(sort_proba)):
            recombined_value = 0
            for j in range(len(self.experts)):
                recombined_value = recombined_value + \
                    self.experts[j].predict_values(np.atleast_2d(x[i]))[0] * sort_proba[i][j]

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
        - extracted, remaining: np.ndarray, np.ndarray
            the extracted values part, the remaining values
        """
        num = values.shape[0]
        indices = np.arange(0, num, quantile) # uniformly distributed
        mask = np.zeros(num, dtype=bool)
        mask[indices] = True
        return values[mask], values[~mask]

    def _find_best_model(self, sorted_trained_values):
        """
        Find the best model which minimizes the errors
        Parameters :
        ------------
        - sorted_trained_values: array_like
        Training samples [[X1,X2, ..., Xn, Y], ... ]
        Optional:
        -----------
        - detail: Boolean
        Set True to see details of the search
        Returns :
        ---------
        - model : Regression_model
        Best model to apply
        - param : dictionary
        Dictionary of its parameters
        - model_name : str
        Name of the model
        """
        dim = self.ndim
        sorted_trained_values = np.array(sorted_trained_values)
        
        rmses = {}
        sms = {}

        # validation with 10% of the training data
        test_values, training_values = self._extract_part(sorted_trained_values, 10)

        for name, sm_class in self._surrogate_type.iteritems():
            if name in ['RMTC', 'RMTB', 'GEKPLS', 'KRG']:
                continue
            
            sm = sm_class()
            sm.options['print_global']=False
            sm.set_training_values(training_values[:, 0:dim], training_values[:, dim])
            sm.train()
            
            expected = test_values[:, dim]
            actual = sm.predict_values(test_values[:, 0:dim])
            l_two = np.linalg.norm(expected - actual, 2)
            l_two_rel = l_two / np.linalg.norm(expected, 2)
            mse = (l_two**2) / len(expected)
            rmse = mse ** 0.5
            rmses[sm.name] = rmse
            print(name, rmse)
            sms[sm.name] = sm
            
        best_name=None
        best_rmse=None
        for name, rmse in rmses.iteritems():
            if best_rmse is None or rmse < best_rmse:
                best_name, best_rmse = name, rmse              
        
        print "BEST = ", best_name
        return sms[best_name]

    def _find_best_heaviside_factor(self, x, y):
        """
        Find the best heaviside factor for smooth approximated values
        Arguments
        ---------
        - x: array_like
        Input training samples
        - y: array_like
        Output training samples

        Returns
        -------
        hfactor : float
        best heaviside factor wrt given samples
        """
        heaviside_factor = 1.
        if self.n_clusters > 1:
            hfactors = np.linspace(0.1, 2.1, num=21)
            errors = []
            for hfactor in hfactors:
                gauss = self._create_multivariate_normal(hfactor)
                predicted_values = self._predict_smooth_output(x, gauss)
                errors.append(Error(y, predicted_values).l_two_rel)
            if max(errors) < 1e-6:
                heaviside_factor = 1.
            else:
                min_error_index = errors.index(min(errors))
                heaviside_factor = hfactors[min_error_index]
        return heaviside_factor
            
    """
    Functions related to clustering
    """
    def _create_multivariate_normal(self, heaviside_factor=1.):
        """
        Create an array of frozen multivariate normal distributions.

        Arguments
        ---------
        - heaviside_factor: float
            Heaviside factor used to scale covariance matrices

        Returns:
        --------
        - gauss_array: array_like
            Array of frozen multivariate normal distributions with means and covariances of the input
        """
        gauss_array = []
        dim= self.ndim
        means = self.cluster.means_
        cov = heaviside_factor*self.cluster.covars_
        for k in range(self.n_clusters):
            meansk = means[k][0:dim]
            covk = cov[k][0:dim, 0:dim]
            rv = sct.multivariate_normal(meansk, covk, True)
            gauss_array.append(rv)
        return gauss_array

    
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

        clustered
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


    @staticmethod
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
            rv = gauss_list[k].pdf(x)
            val = weight[k] * (rv)
            rad = rad + val
            prob.append(val)

        if rad != 0:
            for k in range(len(weight)):
                prob[k] = prob[k] / rad

        clus = prob.index(max(prob))
        return prob, clus

    @staticmethod
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
            v = v + weight[k] * gauss_list[k].pdf(x)
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

    @staticmethod
    def _proba_cluster(dim, weight, gauss_list, x):
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
                proba, cluster = MOE._proba_cluster_one_sample(weight, gauss_list, x[i])
                prob.append(proba)
                clus.append(cluster)

        return np.array(prob), np.array(clus)

    @staticmethod
    def _derive_proba_cluster(dim, weight, gauss_list, x):
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


    ################################################################################
    def plotClusterGMM(self, Max, Min, x_, y_, heaviside=False):
        """
        Plot gaussian cluster
        Parameters:
        -----------
        GMM : mixture.GMM
        Cluster to plot
        Max: array_like
        Maximum for each dimension
        Min: array_like
        Minimum for each dimension
        x_: array_like
        Input training samples
        y_: array_like
        Output training samples
        Optionnals:
        -----------
        heaviside: float
        Heaviside factor. Default to False
        """
        from matplotlib import colors
        import matplotlib.pyplot as plt
        
        GMM=self.cluster

        if GMM.n_components > 1:
            if heaviside == False:
                heaviside = 1

            colors_ = list(six.iteritems(colors.cnames))

            if isinstance(Max, int) or isinstance(Max, np.float64) or isinstance(Max, float):
                dim = 1
            else:
                dim = len(Max)
            weight = GMM.weights_
            mean = GMM.means_
            cov = GMM.covars_
            gauss = self._create_multivariate_normal(heaviside)
            prob_, sort = self._proba_cluster(dim, weight, gauss, x_)
            if dim == 1:
                fig = plt.figure()
                x = np.linspace(Min, Max)
                prob, clus = self._proba_cluster(dim, weight, gauss, x)
                for i in range(len(weight)):
                    plt.plot(x, prob[:, i], ls='--')
                plt.xlabel('Input Values')
                plt.ylabel('Membership probabilities')
                plt.title('Cluster Map')

                fig = plt.figure()
                for i in range(len(sort)):
                    color_ind = int(((len(colors_) - 1) / sort.max()) * sort[i])
                    color = colors_[color_ind][0]
                    plt.plot(x_[i], y_[i], c=color, marker='o')
                plt.xlabel('Input Values')
                plt.ylabel('Output Values')
                plt.title('Samples with clusters')

            if dim == 2:
                x0 = np.linspace(-5, 10, 20)
                x1 = np.linspace(0, 15, 20)
                xv, yv = np.meshgrid(x0, x1)
                x = np.array(zip(xv.reshape((-1,)), yv.reshape((-1,))))
                print(x)
                prob, clus = self._proba_cluster(dim, weight, gauss, x)

                fig = plt.figure()
                ax1 = fig.add_subplot(111, projection='3d')
                for i in range(len(weight)):
                    color = colors_[int(((len(colors_) - 1) / len(weight)) * i)][0]
                    ax1.plot_trisurf(x[:, 0], x[:, 1], prob[:, i], alpha=0.4, linewidth=0,
                                     color=color)
                plt.title('Cluster Map 3D')

                fig1 = plt.figure()
                for i in range(len(weight)):
                    color = colors_[int(((len(colors_) - 1) / len(weight)) * i)][0]
                    plt.tricontour(x[:, 0], x[:, 1], prob[:, i], 1, colors=color, linewidths=3)
                plt.title('Cluster Map 2D')

                fig = plt.figure()
                ax2 = fig.add_subplot(111, projection='3d')
                for i in range(len(sort)):
                    color = colors_[int(((len(colors_) - 1) / sort.max()) * sort[i])][0]
                    ax2.scatter(x_[i][0], x_[i][1], y_[i], c=color)
                plt.title('Samples with clusters')
            plt.show()


class Error(object):
    """
    A class to handle various errors:
    - l_two : float
    L2 error
    - l_two_rel : float
    relative L2 error
    - mse : float
    mse error
    - rmse : float
    rmse error
    - lof : float
    lof error
    - r_two : float
    Residual
    - err_rel: array_like
    relative errors table
    - err_rel_mean : float
    mean of err_rel
    -err_rel_max : float
    max of err_rel
    -err_abs_max: flot
    max of err_abs (norm inf)
    """

    def __init__(self, y_array_true, y_array_calc):
        length = len(y_array_true)
        self.l_two = np.linalg.norm((y_array_true - y_array_calc), 2)
        self.l_two_rel = self.l_two / np.linalg.norm((y_array_true), 2)
        self.mse = (self.l_two**2) / length
        self.rmse = self.mse ** 5
        err = np.abs((y_array_true - y_array_calc) / y_array_true)
        self.err_rel = 100 * err
        self.err_rel_mean = np.mean(self.err_rel)
        self.err_rel_max = max(self.err_rel)
        self.err_abs_max = np.linalg.norm((y_array_true - y_array_calc), np.inf)
        #self.quant = QuantError(err)
        if abs(np.var(y_array_true)) > 1e-10:
            self.lof = 100 * self.mse / np.var(y_array_true)
            self.r_two = (1 - self.lof / 100)
        else:
            self.lof = None
            self.r_two = None

