"""
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.

Mixture of Experts
"""

#TODO : choice of the surrogate model experts to be used
#TODO : add parameters for KRG, RMTB, RMTC, KRG ?
#TODO : add derivative support
#TODO : add variance support
#TODO : support for best number of clusters
#TODO : add factory to get proper surrogate model object
#TODO : implement verbosity 'print_global'
#TODO : documentation

from __future__ import division
import six
import numpy as np
import warnings
from sklearn import mixture, cluster
from scipy.stats import multivariate_normal

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
        Supports for surrogate model API.
        """
        super(MOE, self).apply_method()

    def predict_values(self, x):
        """
        Support for surrogate model API.
        """
        return super(MOE, self).analyse_results(x=x, operation='predict_values')        

    def _apply(self):
        """
        Build and train the mixture of experts surrogate.
        This method is called by Extension apply() method
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
            print('Best Heaviside factor = {}'.format(self.heaviside_factor))
            self.distribs = self._create_clusters_distributions(self.heaviside_factor)

        if not test_data_present:
            # if we have used part of data to validate, fit on overall data
            self._fit(x, y, c, new_model=False)


    def _analyse_results(self, x, operation='predict_values', kx=None):
        """
        Analyse the mixture of experts at the given samples x wrt the specified operation.
        This method is called by Extension analyse_results() method.

        Arguments
        ----------
        x : np.ndarray[n, nx] or np.ndarray[n]
            Input values for the prediction result analysis.

        operation: str
            Type of the analysis. A value is available: 'predict_values'

        kx : int
            The 0-based index of the input variable with respect to which derivatives are desired.

        Return
        ------
        y: np.ndarray
            Output values at the prediction value/derivative points.

        """
        if operation == 'predict_values':
            if self.smooth_recombination:
                y = self._predict_smooth_output(x)
            else:
                y = self._predict_hard_output(x)
            return y 
        else:
            raise ValueError("MOE supports predict_values operation only.")
        return y

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
                self.experts.append(model)
            else:  # retrain the experts with the 
                trained_values = np.array(clusters[i])
                x_trained = trained_values[:, 0:self.ndim]
                y_trained = trained_values[:, self.ndim]
                self.experts[i].set_training_values(x_trained, y_trained)
                self.experts[i].train()
        
    def _predict_hard_output(self, x):
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
            model = self.experts[sort_cluster[i]]
            predicted_values.append(model.predict_values(np.atleast_2d(x[i]))[0])
        predicted_values = np.array(predicted_values)

        return predicted_values

    def _predict_smooth_output(self, x, distribs=None):
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
        - extracted, remaining : np.ndarray, np.ndarray
            the extracted values part, the remaining values

        """
        num = values.shape[0]
        indices = np.arange(0, num, quantile) # uniformly distributed
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
        
        rmses = {}
        sms = {}

        # validation with 10% of the training data
        test_values, training_values = self._extract_part(clustered_values, 10)
        
        for name, sm_class in self._surrogate_type.iteritems():
            if name in ['RMTC', 'RMTB', 'GEKPLS', 'KRG']:  
                # SMs not used for now because require some parameterization
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
            print(name, rmse, mse)
            sms[sm.name] = sm
            
        best_name=None
        best_rmse=None
        for name, rmse in rmses.iteritems():
            if best_rmse is None or rmse < best_rmse:
                best_name, best_rmse = name, rmse              
        
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
        heaviside_factor = 1.
        if self.n_clusters > 1:
            hfactors = np.linspace(0.1, 2.1, num=21)
            errors = []
            for hfactor in hfactors:
                distribs = self._create_clusters_distributions(hfactor)
                ypred = self._predict_smooth_output(x, distribs)
                err_rel = np.linalg.norm(y - ypred, 2) / np.linalg.norm(y, 2)
                errors.append(err_rel)
            if max(errors) < 1e-6:
                heaviside_factor = 1.
            else:
                min_error_index = errors.index(min(errors))
                heaviside_factor = hfactors[min_error_index]
        return heaviside_factor
            
    """
    Functions related to clustering
    """
    def _create_clusters_distributions(self, heaviside_factor=1.):
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
        dim= self.ndim
        means = self.cluster.means_
        cov = heaviside_factor*self.cluster.covars_
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
            probs = probs/rad
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
            probs = np.array([self._proba_cluster_one_sample(x[i], distribs) for i in range(len(x))])

        return probs

