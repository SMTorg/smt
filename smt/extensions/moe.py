"""
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.

Mixture of Experts
"""

from __future__ import division
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

        declare('number_cluster', 2, types=int, desc='Number of cluster')
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

        self.number_cluster = None
        self.smooth_recombination = None
        self.heaviside_optimization = None
        self.heaviside_factor = 1.

        self.experts = None
        self.model_list = []

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

        self.number_cluster = self.options['number_cluster']
        self.smooth_recombination = not self.options['smooth_recombination']
        self.heaviside_optimization = self.options['smooth_recombination'] and self.options['heaviside_optimization']
        self.heaviside_factor = 1.

        self._check_inputs()

        self.expert_types = self._select_expert_types()
        self.model_list = []

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
        self.cluster = mixture.GMM(n_components=self.number_cluster,
                                   covariance_type='full', n_init=20)
        self.cluster.fit(np.c_[xt, ct])        

        # Choice of the experts and training
        self._fit(xt, yt, ct)

        xtest = self.test_values[:, 0:nx]
        ytest = self.test_values[:, nx:nx+1]

        # Heaviside factor
        if self.heaviside_optimization and self.number_cluster > 1:
            self.heaviside_factor = self._find_best_heaviside_factor(xtest, ytest)
            print('BEST HEAVISIDE =', self.heaviside_factor)
            self.gauss = self._create_multivar_normal_dis(self.heaviside_factor)

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
        max_number_cluster = int(len(self.x) / 10) + 1
        if self.number_cluster > max_number_cluster:
            print 'Number of clusters should be inferior to {0}'.format(max_number_cluster)
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
        self.gauss = self._create_multivar_normal_dis(self.heaviside_factor)

        sort_cluster = self.cluster.predict(np.c_[x_trained, c_trained])
        print(sort_cluster)

        # sort trained_values for each cluster
        trained_cluster = self._sort_values_by_cluster(
            np.c_[x_trained, y_trained], self.number_cluster, sort_cluster)

        # find model for each cluster
        for clus in range(self.number_cluster):

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
                    model = self._find_best_model(trained_cluster[clus])

                self.model_list.append(model)

            else:  # Train on the overall domain
                trained_values = np.array(trained_cluster[clus])
                x_trained = trained_values[:, 0:self.ndim]
                y_trained = trained_values[:, self.ndim]
                self.model_list[clus].set_training_values(x_trained, y_trained)
                self.model_list[clus].train()
        
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
        sort_cluster = self._proba_cluster(
            self.ndim, self.cluster.weights_, self.gauss, x)[1]

        for i in range(len(sort_cluster)):
            model = self.model_list[sort_cluster[i]]
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
        g = gauss or self.gauss
        sort_proba, _ = self._proba_cluster(self.ndim, self.cluster.weights_, g, x)

        for i in range(len(sort_proba)):
            recombined_value = 0
            for j in range(len(self.model_list)):
                recombined_value = recombined_value + \
                    self.model_list[j].predict_values(np.atleast_2d(x[i]))[0] * sort_proba[i][j]

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
        print (values.shape[0], values[mask])
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
            
            expected = self.test_values[:, dim]
            actual = sm.predict_values(self.test_values[:, 0:dim])
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
        if self.cluster.n_components > 1:
            heaviside_factors_array = np.linspace(0.1, 2.1, num=21)
            errors_list_by_heaviside_factor = []

            for hfactor in heaviside_factors_array:
                gauss = self._create_multivar_normal_dis(hfactor)
                predicted_values = self._predict_smooth_output(x, gauss)
                errors_list_by_heaviside_factor.append(Error(y, predicted_values).l_two_rel)

            min_error_index = errors_list_by_heaviside_factor.index(
                min(errors_list_by_heaviside_factor))

            if max(errors_list_by_heaviside_factor) < 1e-6:
                heaviside_factor = 1.
            else:
                heaviside_factor = heaviside_factors_array[min_error_index]
        return heaviside_factor

    """
    Functions related to clustering
    """
    def _create_multivar_normal_dis(self, heaviside_factor=1.):
        """
        Create an array of frozen multivariate normal distributions.

        Arguments
        ---------
        - heaviside_factor: float
            Heaviside factor used toscla covariance matrices

        Returns:
        --------
        - gauss_array: array_like
            Array of frozen multivariate normal distributions with means and covariances of the input
        """
        gauss_array = []
        dim= self.ndim
        means = self.cluster.means_
        cov = heaviside_factor*self.cluster.covars_
        for k in range(len(means)):
            meansk = means[k][0:dim]
            covk = cov[k][0:dim, 0:dim]
            rv = sct.multivariate_normal(meansk, covk, True)
            gauss_array.append(rv)
        return gauss_array

    @staticmethod
    def _sort_values_by_cluster(values, number_cluster, sort_cluster):
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
