"""
The main file which handles the mixture of experts
"""
# -*- coding: utf-8 -*-
# Mixture of expert

import warnings
import os
import time
import gc
try:
    import cPickle as pickle
except:
    import pickle
import six

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

import scipy
from scipy.interpolate import Rbf

import numpy as np
from numpy import float64

from smt.mixture.cluster import create_clustering, sort_values_by_cluster, create_multivar_normal_dis, proba_cluster
from smt.mixture.utils import sum_x, cut_list, concat_except
from smt.mixture.error import Error
from smt.mixture.factory import ModelFactory


warnings.filterwarnings("ignore", category=DeprecationWarning)


class MoE(object):
    """
    Create a Mixture of Expert which have the following attributes :
    - x: array_like
        Input training samples
    - y; array_like
        Output Training Samples
    - c: array_like
        Cluster training samples
    - number_cluster:int
        Number of Clusters
    - dimension:int
        Dimension of inputs samples
    - hard: boolean
        Hard or smooth recombination
    - scale_factor:float
        heaviside factor
    - valid_hard: Error
        Error for the discontinuous recombination
    - valid_smooth: Error
        Error for smooth recombination
    - cluster:
        Clustering
    - model_list:array_like
        List of models sort by cluster
    - model_list_name:array_like
        List of model name sort by cluster
    - model_list_param:array_like
        List of model param sort by cluster
    - test_values : array_like
        test_values
    - trained_values : array_like
        trained_values
    - val : array_like
        all values (x, y, c)
    - factory : ModelFactory object
        Contains the available models and parameters
    - gauss :
        Array of frozen multivariate normal distributions
    """

    def __init__(self, number_cluster=None, optim=False, hard_recombination=None, sigma=False):
        """
        Initialize the Moe object
        Optional:
        ----------
        - number_cluster: int
        Number of clusters
        - optim: Boolean
        Set to True to use models with jacobian predictions. Default to False.
        - hard_recombination : Boolean
        Set true to a hard recombination
        - sigma: boolean
        Set to True to use models with sigma predictions. Default to False.
        """
        self.x = None
        self.y = None
        self.c = None
        self.number_cluster = number_cluster
        self.dimension = None
        self.hard = hard_recombination
        self.scale_factor = 1
        self.valid_hard = None
        self.valid_smooth = None
        self.cluster = None
        self.model_list = []
        self.model_list_name = []
        self.model_list_param = []
        self.test_values = None
        self.trained_values = None
        self.val = None
        self.factory = ModelFactory(optim=optim, sigma=sigma)
        self.gauss = None

##########################################################################

    def set_possible_models(self, available_models):
        """
        This method allows to set the different models and their parameters
        Parameters :
        -----------
        - available_models :
        A list of dictionaries which are the different possible models and its parameters
        The key 'type' which indicates the model is compulsory
        For the other parameters, see the documentation or the init of the model in the package models
        """
        self.factory.set_models(available_models)

##########################################################################

    def add_model(self, model):
        """
        This method allows to add a model and its parameters to the mixture
        Parameters :
        -----------
        - model :
        A dictionary which represents a model and its parameters
        The key 'type' which indicates the model is compulsory
        """
        self.factory.add_model(model)

##########################################################################

    def get_possible_models(self):
        """
        This method prints the different available models for the mixture
        """
        available_model = self.factory._get_available_models()
        for model in available_model:
            if model['type'] in ['Krig', 'KrigPLS', 'KrigPLSK']:
                print(model['type'], 'corr=' +
                      model['corr'], 'regr=' + model['regr'])
            elif model['type'] == 'PA':
                print(model['type'], 'degree=' + str(model['degree']))
            elif model['type'] == 'KRGsmt':
                print(model['type'], 'name=' + model['name'])
            else:
                print(model['type'])


##########################################################################

    def _find_best_model(self, sorted_trained_values, detail=False):
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
        dimension = self.dimension
        sorted_trained_values = np.array(sorted_trained_values)

        model_list = []
        error_list_by_model = []
        name_model_list = []
        param_model_list = []

        # Training for each model
        print self.factory.available_models
        for i, params in enumerate(self.factory.available_models):
#             try:
            model, param = self.factory.create_trained_model(
                sorted_trained_values[:, 0:dimension], sorted_trained_values[:, dimension], dimension, params)  # Training
            erreurm = error.Error(self.test_values[:, dimension], model.predict_output(
                self.test_values[:, 0:dimension]))  # errors
            model_list.append(model)
            param_model_list.append(param)
            error_list_by_model.append(erreurm.l_two_rel)
            name_model_list.append(
                self.factory.available_models[i]['type'])

#             except NameError:
#                 if detail:  # pragma: no cover
#                     print self.factory.available_models[i]['type'] + " can not be performed"

        # find minimum error
        min_error_index = error_list_by_model.index(min(error_list_by_model))

        # find best model and train it with all the training samples
        model = model_list[min_error_index]
        param = param_model_list[min_error_index]
        model_name = name_model_list[min_error_index]

        if detail:  # pragma: no cover
            print "Best Model:", model_name
            print "#######"

        return model, param, model_name

##########################################################################

    def fit_without_clustering(self, dimension, x_trained, y_trained, c_trained, detail=False, new_model=True):
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
        self.check_input_data(dimension, x_trained, y_trained, c_trained)

        self.dimension = dimension

        self.gauss = create_multivar_normal_dis(self.dimension, self.cluster.means_,
                                                self.scale_factor * self.cluster.covars_)

        sort_cluster = self.cluster.predict(np.c_[x_trained, c_trained])

        # sort trained_values for each cluster
        trained_cluster = sort_values_by_cluster(
            np.c_[x_trained, y_trained], self.number_cluster, sort_cluster)

        # find model for each cluster
        for clus in range(self.number_cluster):

            if detail:  # pragma: no cover
                print "Number of cluster:", clus + 1

            if new_model:

                if len(self.factory.available_models) == 1:
                    x_trained = np.array(trained_cluster[clus])[:, 0:dimension]
                    y_trained = np.array(trained_cluster[clus])[:, dimension]
                    model, param = self.factory.create_trained_model(
                        x_trained, y_trained, dimension, self.factory.available_models[0])
                    model_name = self.factory.available_models[0]['type']

                else:
                    model, param, model_name = self._find_best_model(
                        trained_cluster[clus], detail)

                self.model_list.append(model)
                self.model_list_name.append(model_name)
                self.model_list_param.append(param)

            else:  # Train on the overall domain
                trained_values = np.array(trained_cluster[clus])
                x_trained = trained_values[:, 0:dimension]
                y_trained = trained_values[:, dimension]
                self.model_list[clus].train(x_trained, y_trained)

        gc.collect()

##########################################################################

    def _best_scale_factor(self, x, y, detail=None, plot=None):
        """
        Find the best heaviside factor for smooth approximated values
        Parameters:
        -----------
        - x: array_like
        Input training samples
        - y: array_like
        Output training samples
        Optional:
        -----------
        - detail: boolean
        Show details of the search if set to True
        - plot: boolean
        Plot the search if set to True
        """
        if self.cluster.n_components == 1:

            self.scale_factor = 1
        else:

            time1 = time.clock()

            if detail:  # pragma: no cover
                print "SCALE FACTOR OPTIMISATION"
                print "-------------------------"

            scale_factors_array = np.linspace(0.1, 2.1, num=21)
            errors_list_by_scale_factor = []

            for i in scale_factors_array:

                self.gauss = create_multivar_normal_dis(self.dimension, self.cluster.means_,
                                                            i * self.cluster.covars_)
                predicted_values = self._predict_smooth_output(x)
                errors_list_by_scale_factor.append(
                    error.Error(y, predicted_values).l_two_rel)

            min_error_index = errors_list_by_scale_factor.index(
                min(errors_list_by_scale_factor))

            if max(errors_list_by_scale_factor) < 1e-6:
                scale_factor = 1

            else:
                scale_factor = scale_factors_array[min_error_index]

            self.scale_factor = scale_factor

            time2 = time.clock()

            if detail:  # pragma: no cover
                print "Best scale_factor factor:", scale_factor
                print self.format_time(time2 - time1)
                print "#######"

            if plot:  # pragma: no cover
                fig = plt.figure()
                plt.plot(scale_factors_array,
                         errors_list_by_scale_factor, 'ro', linestyle='-')
                plt.xlabel('Heaviside Factor')
                plt.ylabel('Error')
                plt.title('Evolution of Error with Heaviside Factor')
                plt.show()

###################################################################

    def fit(self, dimension, x, y, c=None, valid=None, detail=False, plot=False, heaviside=False, number_cluster=0,
            median=True, max_number_cluster=None, cluster_model=None, reset=True):
        """
        Train the MoE with 90 percent of the input samples if valid is None
        The last 10 percent are used to valid the model
        Parameters :
        ------------
        - dimension: int
        Dimension of input samples
        - x: array_like
        Input training samples
        - y:array_like
        Output training samples
        Optional :
        ------------
        - c: array_like
        Clustering training samples. Default to None. In this case, c=y.
        - valid : array_like
        Validating samples if they exist. Default to None (10%)
        - detail: Boolean
        Set True to see details of the Moe creation
        - plot: Boolean
        Set True to see comparison between real and predicted values
        - heaviside: Boolean
        Set True to optimize the heaviside factor
        - number_cluster:int
        Number of clusters. Default to 0 to find the best number of clusters
        - median:boolean
        Set True to find the best number of clusters with the median criteria.
        Set False to find the best number of clusters with means of criteria
        Default to True.
        - max_number_cluster:int
        Maximum number of clusters. Default to None (tenth of the length of x)
        - cluster_model: func
        Model used to find the best number of cluster
        - reset: Boolean
        Reset the best model list and the clustering if True. Default to True
        """
        if c is None:
            c = y

        self.check_input_data(dimension, x, y, c)

        x = np.array(x)
        y = np.array(y)
        c = np.array(c)

        self.dimension = dimension
        self.x = x
        self.y = y
        self.c = c

        if reset:
            self.model_list = []
            self.model_list_name = []
            self.model_list_param = []

        time1 = time.clock()

        # choice of test values and trained values
        values = np.c_[x, y, c]

        if valid is None:
            cut_values = cut_list(values, 10)  # cut space in 10 parts
            trained_values = np.vstack(concat_except(
                cut_values, 0))  # assemble 9 parts
            test_values = np.vstack(cut_values[0])
        else:
            trained_values = values
            test_values = valid

        self.test_values = test_values
        self.trained_values = trained_values
        self.val = values

        total_length = len(trained_values[0])
        x_trained = trained_values[:, 0:dimension]
        y_trained = trained_values[:, dimension]
        c_trained = trained_values[:, dimension + 1:total_length]

        # choice of number of cluster
        if reset:
            if max_number_cluster is None:
                max_number_cluster = int(len(x) / 10) + 1

            if number_cluster is 0:
                self._default_number_cluster(dimension, x, y, c, detail=detail, plot=plot,
                                             median=median, max_number_cluster=max_number_cluster, available_models=cluster_model)
            else:
                self.number_cluster = number_cluster

            if self.number_cluster > max_number_cluster:
                print 'Number of clusters should be inferior to {0}'.format(max_number_cluster)
                raise ValueError(
                    'The number of clusters is too high considering the number of points')

        time2 = time.clock()

        if detail:  # pragma: no cover
            print "##############################################"
            print "EM CLUSTERING PERFORMED"
            print "-----------------------"

        # Cluster space
        if reset:
            self.cluster = create_clustering(
                x_trained, c_trained, n_component=self.number_cluster, method='GMM')
            while self._check_number_cluster(x_trained, y_trained, c_trained) is False:
                self.number_cluster = self.number_cluster - 1
                warnings.warn(
                    "The number of cluster had to be reduced in order to have enough points by cluster")
                self.cluster = create_clustering(
                    x_trained, c_trained, n_component=self.number_cluster, method='GMM')
                print 'The new number of clusters is {0}'.format(self.number_cluster)

        time3 = time.clock()

        if detail:  # pragma: no cover
            print "Computation time (clustering): ", self.format_time(time3 - time2)
            print "##############################################"
            print "LOCAL BEST MODELING PERFORMED"
            print "-----------------------------"

        # Choice of the models and training
        self.fit_without_clustering(
            dimension, x_trained, y_trained, c_trained, detail=detail, new_model=reset)

        time4 = time.clock()

        if detail:  # pragma: no cover
            print "Computation time (model): ", self.format_time(time4 - time3)
            print "##############################################"

        # choice of heaviside factor
        if heaviside and self.number_cluster > 1:
            self._best_scale_factor(
                test_values[:, 0:dimension], test_values[:, dimension], detail=detail, plot=plot)

        self.gauss = create_multivar_normal_dis(self.dimension, self.cluster.means_,
                                                    self.scale_factor * self.cluster.covars_)

        time5 = time.clock()

        # Validation
        self._valid(test_values[:, 0:dimension], test_values[:, dimension])

        if self.valid_hard.l_two_rel > self.valid_smooth.l_two_rel:
            if self.hard is None:
                self.hard = False
        else:
            if self.hard is None:
                self.hard = True

        # once the validation is done, the model is trained again on all the
        # space
        if valid is None:
            self.fit_without_clustering(dimension, x, y, c, new_model=False)

        time6 = time.clock()

        if detail:  # pragma: no cover
            print "##############################################"
            print "VALIDATION:"
            print "-----------"
            print "Error (hard):", self.valid_hard.l_two_rel
            print "Error (smooth):", self.valid_smooth.l_two_rel
            print "Computation time (Validation): ", self.format_time(time6 - time5)
            print "Computation time (All): ", self.format_time(time6 - time1)
            print "##############################################"

        gc.collect()

###################################################################

    def _check_number_cluster(self, x_trained, y_trained, c_trained):
        """
        This function checks the number of cluster isn't too high
        so that it will allow to have enough points by cluster.
        Parameters:
        -----------
        - x_trained: array_like
        input sample trained
        - y_trained: array_like
        output sample trained
        - c_trained: array_like
        input sample weights trained
        Return:
        -----------
        - ok: boolean
        True if the number of cluster is ok else False
        """

        ok = True
        sort_cluster = self.cluster.predict(np.c_[x_trained, c_trained])
        trained_cluster = sort_values_by_cluster(
            np.c_[x_trained, y_trained], self.number_cluster, sort_cluster)

        for model in self.factory.available_models:

            if model['type'] == 'PA':
                min_number_point = np.math.factorial(self.dimension + model['degree']) / (
                    np.math.factorial(self.dimension) * np.math.factorial(model['degree']))
                ok = self._check_number_points(
                    trained_cluster, min_number_point)

            if model['type'] == 'PA2' or model['type'] == 'PA2smt':
                min_number_point = np.math.factorial(self.dimension + 2) / (
                    np.math.factorial(self.dimension) * np.math.factorial(2))
                ok = self._check_number_points(
                    trained_cluster, min_number_point)

            if model['type'] == 'LS':
                min_number_point = 1 + self.dimension
                ok = self._check_number_points(
                    trained_cluster, min_number_point)

            if model['type'] in ['Krig', 'KrigPLS', 'KrigPLSK']:

                if model['regr'] == 'linear':
                    min_number_point = 1 + self.dimension
                    ok = self._check_number_points(
                        trained_cluster, min_number_point)

                if model['regr'] == 'quadratic':
                    min_number_point = np.math.factorial(self.dimension + 2) / (
                        np.math.factorial(self.dimension) * np.math.factorial(2))
                    ok = self._check_number_points(
                        trained_cluster, min_number_point)

            if ok is False:
                return False

        return True

###################################################################

    def _check_number_points(self, trained_cluster, min_number_point):
        """
        This function checks the number of points is enough high
        considering the minimum number required by the model
        Parameters:
        -----------
        - trained_cluster: array_like
        values of trained sample sorted by cluster
        - min_number_point: int
        the minimum number of points required
        Return:
        -----------
        boolean:
        True if it's ok else False
        """
        for array in trained_cluster:
            if len(array) < min_number_point:
                return False
        return True

###################################################################

    def _default_number_cluster(self, dim, x, y, c, plot=False, detail=False, available_models=None, max_number_cluster=None, median=True):
        """
        Find the best number of clusters thanks to a cross-validation.
        Parameters :
        ------------
        - dim: int
        Dimension of the problem
        - x: array_like
        Inputs Training values
        - y: array_like
        Target Values
        - c:array_like
        Clustering Criterion
        Optional :
        ------------
        - plot: Boolean
        Set True to plot cross-validation evolution and box plots
        - detail: Boolean
        Set True to see detail of clustering
        - available_models: dictionary
        Model to apply for each cluster. Default to PA2 if dim <=9 else RadBF (arbitrary choice)
        - max_number_cluster:int
        The maximal number of clusters. Default to tenth of the X_train length
        - median: boolean
        Find the best number of clusters with the median of cross-validations set to True.
        Find the best number of clusters with the MSE of cross-validations set to False.
        Returns :
        ----------
        - cluster: int
        The best number of clusters 'by default'
        """
        tps_dep = time.clock()

        # Detail input
        if detail:  # pragma: no cover
            print "BEST NUMBER OF CLUSTER"
            print "----------------------"
            print "INPUT: "
            print "Dimension: ", dim
            print "Number of samples: ", len(x)
            print "#######"

        val = np.c_[x, y, c]
        total_length = len(val[0])
        val_cut = cut_list(val, 5)

        # Stock
        errori = []
        erroris = []
        posi = []
        b_ic = []
        a_ic = []
        error_h = []
        error_s = []
        median_eh = []
        median_es = []

        # Init Output Loop
        auxkh = 0
        auxkph = 0
        auxkpph = 0
        auxks = 0
        auxkps = 0
        auxkpps = 0
        ok1 = True
        ok2 = True
        i = 0
        exit_ = True

        # Find error for each cluster
        while i < max_number_cluster and exit_:
            if available_models is None:
                if dim > 9:
                    available_models = [{'type': 'RadBF'}]
                else:
                    available_models = [{'type': 'PA', 'degree': 2}]

            # Stock
            errorc = []
            errors = []
            bic_c = []
            aic_c = []
            ok = True  # Say if this number of cluster is possible

            # Cross Validation
            for c in range(5):
                if ok:
                    # Create training and test samples
                    val_train = np.vstack(concat_except(val_cut, c))
                    val_test = np.vstack(val_cut[c])
                    # Create the MoE for the cross validation
                    mixture = MoE(i + 1)
                    mixture.set_possible_models(available_models)
                    mixture.cluster = create_clustering(
                        val_train[:, 0:dim], val_train[:, dim + 1:total_length], i + 1, 'GMM')  # Clustering
                    valc = np.c_[
                        val_train[:, 0:dim], val_train[:, dim + 1:total_length]]
                    sort = mixture.cluster.predict(valc)
                    clus_train = sort_values_by_cluster(
                        val_train[:, 0:dim + 1], i + 1, sort)
                    bic_c.append(
                        mixture.cluster.bic(np.c_[val_test[:, 0:dim], val_test[:, dim + 1:total_length]]))
                    aic_c.append(
                        mixture.cluster.aic(np.c_[val_test[:, 0:dim], val_test[:, dim + 1:total_length]]))
                    for j in range(i + 1):
                        # If there is at least one point
                        if len(clus_train[j]) < 4:
                            ok = False

                    if ok:
                        # calculate error
                        try:
                            # Train the MoE for the cross validation
                            mixture.fit_without_clustering(dim, val_train[:, 0:dim], val_train[:, dim],
                                                           val_train[:, dim + 1:total_length])
                            # errors Of the MoE
                            errorcc = error.Error(val_test[:, dim],
                                                  mixture._predict_hard_output(val_test[:, 0:dim]))
                            errorsc = error.Error(val_test[:, dim],
                                                  mixture._predict_smooth_output(val_test[:, 0:dim]))
                            errors.append(errorsc.l_two_rel)
                            errorc.append(errorcc.l_two_rel)
                        except:
                            errorc.append(1.)  # extrem value
                            errors.append(1.)
                    else:
                        errorc.append(1.)  # extrem value
                        errors.append(1.)

            # Stock for box plot
            b_ic.append(bic_c)
            a_ic.append(aic_c)
            error_s.append(errors)
            error_h.append(errorc)

            # Stock median
            median_eh.append(np.median(errorc))
            median_es.append(np.median(errors))

            # Stock possible numbers of cluster
            if ok:
                posi.append(i)

            # Stock mean errors
            errori.append(np.mean(errorc))
            erroris.append(np.mean(errors))

            if detail:  # pragma: no cover
                # Print details about the clustering
                print 'Number of cluster:', i + 1, '| Possible:', ok, '| Error(hard):', np.mean(errorc), '| Error(smooth):', np.mean(errors), '| Median (hard):', np.median(errorc), '| Median (smooth):', np.median(errors)
                print '#######'

            if i > 3:
                # Stop the search if the clustering can not be performed three
                # times
                ok2 = ok1
                ok1 = ok
                if ok1 is False and ok is False and ok2 is False:
                    exit_ = False
            if median:
                # Stop the search if the median increases three times
                if i > 3:
                    auxkh = median_eh[i - 2]
                    auxkph = median_eh[i - 1]
                    auxkpph = median_eh[i]
                    auxks = median_es[i - 2]
                    auxkps = median_es[i - 1]
                    auxkpps = median_es[i]

                    if auxkph >= auxkh and auxkps >= auxks and auxkpph >= auxkph and auxkpps >= auxkps:
                        exit_ = False
            else:
                if i > 3:
                    # Stop the search if the means of errors increase three
                    # times
                    auxkh = errori[i - 2]
                    auxkph = errori[i - 1]
                    auxkpph = errori[i]
                    auxks = erroris[i - 2]
                    auxkps = erroris[i - 1]
                    auxkpps = erroris[i]

                    if auxkph >= auxkh and auxkps >= auxks and auxkpph >= auxkph and auxkpps >= auxkps:
                        exit_ = False
            i = i + 1

        # Find The best number of cluster
        cluster_mse = 1
        cluster_mses = 1
        if median:
            min_err = median_eh[posi[0]]
            min_errs = median_es[posi[0]]
        else:
            min_err = errori[posi[0]]
            min_errs = erroris[posi[0]]
        for k in posi:
            if median:
                if min_err > median_eh[k]:
                    min_err = median_eh[k]
                    cluster_mse = k + 1
                if min_errs > median_es[k]:
                    min_errs = median_es[k]
                    cluster_mses = k + 1
            else:
                if min_err > errori[k]:
                    min_err = errori[k]
                    cluster_mse = k + 1
                if min_errs > erroris[k]:
                    min_errs = erroris[k]
                    cluster_mses = k + 1

        # Choose between hard or smooth recombination
        if median:
            if median_eh[cluster_mse - 1] < median_es[cluster_mses - 1]:
                cluster = cluster_mse
                hardi = True

            else:
                cluster = cluster_mses
                hardi = False
        else:
            if errori[cluster_mse - 1] < erroris[cluster_mses - 1]:
                cluster = cluster_mse
                hardi = True
            else:
                cluster = cluster_mses
                hardi = False

        self.number_cluster = cluster

        if plot:
            self._plot_number_cluster(
                posi, errori, erroris, median_eh, median_es, b_ic, a_ic, error_h, error_s, cluster)

        if median:
            method = '| Method: Minimum of Median errors'
        else:
            method = '| Method: Minimum of relative L2'

        tps_fin = time.clock()

        if detail:  # pragma: no cover
            print 'Optimal Number of cluster: ', cluster, method
            print 'Recombination Hard: ', hardi
            print 'Computation time (cluster): ', self.format_time(tps_fin - tps_dep)


###################################################################

    def predict_output(self, x):
        """
        This method predicts the output of a x samples
        Parameters:
        ----------
        - x: Array_like
        x samples
        Return :
        ----------
        - predicted_values : array_like
        predicted output
        """

        if self.hard:
            predicted_values = self._predict_hard_output(x)
            return predicted_values

        else:
            predicted_values = self._predict_smooth_output(x)
            return predicted_values

###################################################################

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
        sort_cluster = proba_cluster(
            self.dimension, self.cluster.weights_, self.gauss, x)[1]

        for i in range(len(sort_cluster)):
            model = self.model_list[sort_cluster[i]]
            predicted_values.append(model.predict_output(x[i])[0])
        predicted_values = np.array(predicted_values)

        return predicted_values

###################################################################

    def _predict_smooth_output(self, x):
        """
        This method predicts the output of a x samples for a smooth recombination
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
        sort_proba = proba_cluster(
            self.dimension, self.cluster.weights_, self.gauss, x)[0]
        for i in range(len(sort_proba)):
            recombined_value = 0

            for j in range(len(self.model_list)):
                recombined_value = recombined_value + \
                    self.model_list[j].predict_output(
                        x[i])[0] * sort_proba[i][j]

            predicted_values.append(recombined_value)

        predicted_values = np.array(predicted_values)

        return predicted_values

###################################################################

    def predict_variance(self, x):
        """
        This method predicts the variance of a x samples
        Parameters:
        ----------
        - x: Array_like
        x samples
        Return :
        ----------
        - predicted_variance : array_like
        predicted variance
        """

        if self.hard:
            predicted_variance = self._predict_hard_variance(x)[1]
            return predicted_variance

        else:
            predicted_variance = self._predict_smooth_variance(x)[1]
            return predicted_variance

###################################################################

    def _predict_hard_variance(self, x):
        """
        This method predicts the variance and the output of a x samples for a hard recombination
        Parameters:
        ----------
        - x: Array_like
        x samples
        Return :
        ----------
        - predicted_values : array_like
        predicted output
        - predicted_variance : array_like
        predicted variance
        """
        predicted_values = []
        predicted_variance = []
        sort_cluster = cls.proba_cluster(
            self.dimension, self.cluster.weights_, self.gauss, x)[1]

        for model in self.model_list:
            if model.existing_sigma is False:
                raise ValueError("Models have not variance prediction")

        for i in range(len(sort_cluster)):
            model = self.model_list[sort_cluster[i]]
            predicted_value, predicted_sigma = model.predict_output(
                x[i], eval_MSE=True)
            predicted_values.append(predicted_value[0])
            predicted_variance.append(predicted_sigma[0])

        predicted_values = np.array(predicted_values)
        predicted_variance = np.array(predicted_variance)

        return predicted_values, predicted_variance

###################################################################

    def _predict_smooth_variance(self, x):
        """
        This method predicts the variance and the output of a x samples for a smooth recombination
        Parameters:
        ----------
        - x: Array_like
        x samples
        Return :
        ----------
        - predicted_values : array_like
        predicted output
        - predicted_variance : array_like
        predicted variance
        """
        for model in self.model_list:
            if model.existing_sigma is False:
                raise ValueError("Models have not variance prediction")

        predicted_values = []
        predicted_variance = []
        sort_proba = cls.proba_cluster(
            self.dimension, self.cluster.weights_, self.gauss, x)[0]

        for i in range(len(sort_proba)):
            recombined_value = 0
            recombined_sigma = 0

            for j in range(len(self.model_list)):
                predicted_value, predicted_sigma = self.model_list[j].predict_output(
                    x[i], eval_MSE=True)
                recombined_value = recombined_value + \
                    predicted_value[0] * sort_proba[i][j]
                recombined_sigma = recombined_sigma + \
                    predicted_sigma[0] * (sort_proba[i][j]) ** 2

            predicted_values.append(recombined_value)
            predicted_variance.append(recombined_sigma)

        predicted_values = np.array(predicted_values)
        predicted_variance = np.array(predicted_variance)

        return predicted_values, predicted_variance

###################################################################

    def predict_jacobian(self, x):
        """
        This method predicts the jacobian of the output of a x samples
        Parameters:
        ----------
        - x: Array_like
        x samples
        Return :
        ----------
        - predicted_jacobian : array_like
        predicted jacobian
        """

        sort_proba_test, sort_test = cls.proba_cluster(
            self.dimension, self.cluster.weights_, self.gauss, x)

        if self.hard:
            predicted_jacobian = self._predict_hard_jacobian(sort_test, x)
            return predicted_jacobian

        else:
            predicted_jacobian = self._predict_smooth_jacobian(
                sort_proba_test, x)
            return predicted_jacobian

###################################################################

    def _predict_hard_jacobian(self, sort_cluster, x):
        """
        This method predicts the jacobian of the output of a x samples for a hard recombination
        Parameters:
        ----------
        - x: Array_like
        x samples
        - sort_cluster: array_like
        Membership to one cluster for each sample
        Return :
        ----------
        - predicted_jacobian : array_like
        predicted jacobian
        """
        predicted_jacobian = []

        for i in range(len(sort_cluster)):
            model = self.model_list[sort_cluster[i]]
            predicted_jacobian.append(model.predict_jacobian(x[i])[0])

        predicted_jacobian = np.array(predicted_jacobian)

        return predicted_jacobian

###################################################################

    def _predict_smooth_jacobian(self, sort_proba, x):
        """
        This method predicts the jacobian of the output of a x samples for a smooth recombination
        Parameters:
        ----------
        - x: Array_like
        x samples
        - sort_proba: array_like
        Membership probabilities to each cluster for each sample
        Return :
        ----------
        - predicted_jacobian : array_like
        predicted jacobian
        """
        for model in self.model_list:
            if model.existing_jacobian is False:
                raise ValueError("Models have not jacobians prediction")

        der_proba_test = cls.derive_proba_cluster(
            self.dimension, self.cluster.weights_, self.gauss, x)

        predicted_jacobian = []
        for i in range(len(sort_proba)):
            recombined_jacobian = 0

            for j in range(len(self.model_list)):
                predicted_value = self.model_list[j].predict_jacobian(x[i])[0]
                recombined_jacobian = recombined_jacobian + \
                    predicted_value * \
                    sort_proba[i][j] + der_proba_test[i][j] * \
                    self.model_list[j].predict_output(x[i])[0]
            predicted_jacobian.append(recombined_jacobian)

        predicted_jacobian = np.array(predicted_jacobian)

        return predicted_jacobian

#######################################################################

    def predict_derive_variance(self, x):
        """
        This method predicts the derive of variance of a x samples
        Parameters:
        ----------
        - x: Array_like
        x samples
        Return :
        ----------
        - predicted_derive_variance : array_like
        predicted derive variance
        """

        sort_proba_test, sort_test = cls.proba_cluster(
            self.dimension, self.cluster.weights_, self.gauss, x)

        if self.hard:
            predicted_derive_variance = self._predict_hard_derive_variance(
                sort_test, x)
            return predicted_derive_variance

        else:
            predicted_derive_variance = self._predict_smooth_derive_variance(
                sort_proba_test, x)
            return predicted_derive_variance

#######################################################################

    def _predict_hard_derive_variance(self, sort_cluster, x):
        """
        This method predicts the derive of variance of a x samples for a hard recombination
        Parameters:
        ----------
        - x: Array_like
        x samples
        - sort_cluster: array_like
        Membership to one cluster for each sample
        Return :
        ----------
        - derive_sigma : array_like
        predicted derive variance
        """
        derive_variance = []

        for model in self.model_list:
            if model.existing_sigma is False:
                raise ValueError("Models have not variance prediction")

        for i in range(len(sort_cluster)):
            model = self.model_list[sort_cluster[i]]
            predicted_derive_variance = model.predict_derived_variance(x[i])[0]
            derive_variance.append(predicted_derive_variance)

        derive_variance = np.array(derive_variance)

        return derive_variance

###################################################################
    def _predict_smooth_derive_variance(self, sort_proba, x):
        """
        This method predicts the derive of variance of a x samples for a smooth recombination
        Parameters:
        ----------
        - x: Array_like
        x samples
        - sort_proba: array_like
        Membership probabilities to each cluster for each sample
        Return :
        ----------
        - derive_variance : array_like
        predicted derive variance
        """
        for model in self.model_list:
            if model.existing_sigma is False:
                raise ValueError("Models have not variance prediction")

        der_proba_test = cls.derive_proba_cluster(
            self.dimension, self.cluster.weights_, self.gauss, x)

        derive_variance = []

        for i in range(len(sort_proba)):
            derive_var = 0
            for j in range(len(self.model_list)):
                predicted_value = self.model_list[j].predict_derived_variance(
                    x[i])[0]
                derive_var = derive_var + predicted_value * \
                    (sort_proba[i][j])**2 + 2 * sort_proba[i][j] * \
                    self.model_list[j].predict_output(
                        x[i], eval_MSE=True)[1][0] * der_proba_test[i][j]
            derive_variance.append(derive_var)
        derive_variance = np.array(derive_variance)

        return derive_variance


#######################################################################

    def _valid(self, x, y):
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

###################################################################

    def plot(self):  # pragma: no cover
        """
        This function plots different aspects of the mixture
        """
        dimension = self.dimension
        test_values = self.test_values
        values = self.val
        # Plot Comparison
        fig = plt.figure()
        plt.subplot(121)
        predict, = plt.plot(test_values[:, dimension], self._predict_hard_output(
            test_values[:, 0:dimension]), 'b+')
        plt.plot(test_values[:, dimension], test_values[:, dimension])
        plt.xlabel('Tested outputs values')
        plt.ylabel('Predicted outputs values')
        plt.title('Hard comparison output')
        plt.subplot(122)
        predict, = plt.plot(test_values[:, dimension],
                            self._predict_smooth_output(test_values[:, 0:dimension]), 'b+')
        plt.plot(test_values[:, dimension], test_values[:, dimension])
        plt.xlabel('Tested outputs values')
        plt.ylabel('Predicted outputs values')
        plt.title('Smooth comparison output')
        plt.show()

        # Culster map
        if dimension < 3 and dimension > 0:
            if dimension == 1:
                maxi = max(values[:, 0])
                mini = min(values[:, 0])
            if dimension == 2:
                maxi = np.zeros(2)
                mini = np.zeros(2)
                maxi[0] = max(values[:, 0])
                maxi[1] = max(values[:, 1])
                mini[0] = min(values[:, 0])
                mini[1] = min(values[:, 1])

            self._plot_cluster_gmm(self.cluster, maxi, mini,
                                   self.x, self.y, heaviside=self.scale_factor)

            fig = plt.figure()
            plt.subplot(121)
            true, = plt.plot(
                range(len(values[:, dimension])), values[:, dimension], 'ro')
            predict, = plt.plot(range(len(values[:, dimension])),
                                self._predict_hard_output(values[:, 0:dimension]), 'b+')
            plt.xlabel('Number of samples')
            plt.ylabel('Value')
            plt.title('Hard comparison output')
            plt.legend(
                [predict, true], ['approximated value - hard', 'Real value'])
            plt.subplot(122)
            true, = plt.plot(
                range(len(values[:, dimension])), values[:, dimension], 'ro')
            predict, = plt.plot(range(len(values[:, dimension])),
                                self._predict_smooth_output(values[:, 0:dimension]), 'b+')
            plt.xlabel('Number of samples')
            plt.ylabel('Value')
            plt.title('Smooth comparison output')
            plt.legend(
                [predict, true], ['approximated value - smooth', 'Real value'])
            plt.show()

###################################################################

    def to_txt(self, name_folder, name_file):  # pragma: no cover
        """
        Create a txt file with all the information about the Mixture of experts
        Parameters:
        -----------
        - name_folder: str
        The path to the file
        - name_file: str
        The file name
        """
        script = name_folder + name_file + ".txt"
        if os.path.exists(script):
            os.remove(script)
        script_moe = open(script, "w")
        script_moe.write('MOE Attributes' + ' \n')
        script_moe.write('--------------' + ' \n')
        script_moe.write('Number of Cluster = ' +
                         str(self.number_cluster) + ' \n')
        script_moe.write('Model Used        = ' +
                         str(self.model_list_name) + ' \n')
        script_moe.write(
            'Scale Factor      = ' + str(self.scale_factor) + ' \n')
        script_moe.write('Recombination Hard= ' + str(self.hard) + ' \n')
        script_moe.write(' \n')
        script_moe.write('Validation Smooth' + ' \n')
        script_moe.write('-----------------' + ' \n')
        script_moe.write('l_two          = ' +
                         str(self.valid_smooth.l_two) + ' \n')
        script_moe.write(
            'Relative l_two = ' + str(self.valid_smooth.l_two_rel) + ' \n')
        script_moe.write('RMSE         = ' +
                         str(self.valid_smooth.rmse) + ' \n')
        script_moe.write('lof         = ' + str(self.valid_smooth.lof) + ' \n')
        script_moe.write('r_two          = ' +
                         str(self.valid_smooth.r_two) + ' \n')
        script_moe.write(
            'Mean of Relative Errors    = ' + str(self.valid_smooth.err_rel_mean) + ' \n')
        script_moe.write(
            'Max of Relative Errors     = ' + str(self.valid_smooth.err_rel_max) + ' \n')
        # modif NB abs error
        script_moe.write(
            'Max of Abs Errors     = ' + str(self.valid_smooth.err_abs_max) + ' \n')
        script_moe.write(
            '50% Relative Error Value   = ' + str(self.valid_smooth.quant.val_50) + ' \n')
        script_moe.write(
            '80% Relative Error Value   = ' + str(self.valid_smooth.quant.val_80) + ' \n')
        script_moe.write(
            '90% Relative Error Value   = ' + str(self.valid_smooth.quant.val_90) + ' \n')
        script_moe.write(
            '95% Relative Error Value   = ' + str(self.valid_smooth.quant.val_95) + ' \n')
        script_moe.write(
            '99% Relative Error Value   = ' + str(self.valid_smooth.quant.val_99) + ' \n')
        script_moe.write(
            '99.9% Relative Error Value = ' + str(self.valid_smooth.quant.val_999) + ' \n')
        script_moe.write('Proportion of Samples With Error Lower 0.5   = ' +
                         str(self.valid_smooth.quant.pro_50) + ' \n')
        script_moe.write('Proportion of Samples With Error Lower 0.2   = ' +
                         str(self.valid_smooth.quant.pro_80) + ' \n')
        script_moe.write('Proportion of Samples With Error Lower 0.1   = ' +
                         str(self.valid_smooth.quant.pro_90) + ' \n')
        script_moe.write('Proportion of Samples With Error Lower 0.05  = ' +
                         str(self.valid_smooth.quant.pro_95) + ' \n')
        script_moe.write('Proportion of Samples With Error Lower 0.01  = ' +
                         str(self.valid_smooth.quant.pro_99) + ' \n')
        script_moe.write('Proportion of Samples With Error Lower 0.001 = ' +
                         str(self.valid_smooth.quant.pro_999) + ' \n')
        script_moe.write(' \n')
        script_moe.write('Validation Discontinuous' + ' \n')
        script_moe.write('------------------------' + ' \n')
        script_moe.write('l_two          = ' +
                         str(self.valid_hard.l_two) + ' \n')
        script_moe.write('Relative l_two = ' +
                         str(self.valid_hard.l_two_rel) + ' \n')
        script_moe.write('RMSE         = ' +
                         str(self.valid_hard.rmse) + ' \n')
        script_moe.write('lof         = ' + str(self.valid_hard.lof) + ' \n')
        script_moe.write('r_two          = ' +
                         str(self.valid_hard.r_two) + ' \n')
        script_moe.write(
            'Mean of Relative Errors    = ' + str(self.valid_hard.err_rel_mean) + ' \n')
        script_moe.write(
            'Max of Relative Errors     = ' + str(self.valid_hard.err_rel_max) + ' \n')
        # modif NB abs error
        script_moe.write(
            'Max of abs Errors     = ' + str(self.valid_hard.err_abs_max) + ' \n')
        script_moe.write(
            '50% Relative Error Value   = ' + str(self.valid_hard.quant.val_50) + ' \n')
        script_moe.write(
            '80% Relative Error Value   = ' + str(self.valid_hard.quant.val_80) + ' \n')
        script_moe.write(
            '90% Relative Error Value   = ' + str(self.valid_hard.quant.val_90) + ' \n')
        script_moe.write(
            '95% Relative Error Value   = ' + str(self.valid_hard.quant.val_95) + ' \n')
        script_moe.write(
            '99% Relative Error Value   = ' + str(self.valid_hard.quant.val_99) + ' \n')
        script_moe.write(
            '99.9% Relative Error Value = ' + str(self.valid_hard.quant.val_999) + ' \n')
        script_moe.write('Proportion of Samples With Error Lower 0.5   = ' +
                         str(self.valid_hard.quant.pro_50) + ' \n')
        script_moe.write('Proportion of Samples With Error Lower 0.2   = ' +
                         str(self.valid_hard.quant.pro_80) + ' \n')
        script_moe.write('Proportion of Samples With Error Lower 0.1   = ' +
                         str(self.valid_hard.quant.pro_90) + ' \n')
        script_moe.write('Proportion of Samples With Error Lower 0.05  = ' +
                         str(self.valid_hard.quant.pro_95) + ' \n')
        script_moe.write('Proportion of Samples With Error Lower 0.01  = ' +
                         str(self.valid_hard.quant.pro_99) + ' \n')
        script_moe.write('Proportion of Samples With Error Lower 0.001 = ' +
                         str(self.valid_hard.quant.pro_999) + ' \n')
        script_moe.close()

#######################################################################

    def _plot_cluster_gmm(self, gmm, maxi, mini, x_, y_, heaviside=False):  # pragma: no cover
        """
        Plot gaussian cluster
        Parameters:
        -----------
        - gmm : mixture.gmm
        Cluster to plot
        - maxi: array_like
        Maximum for each dimension
        - mini: array_like
        Minimum for each dimension
        - x_: array_like
        Input training samples
        - y_: array_like
        Output training samples
        Optional:
        -----------
        - heaviside: boolean
        Heaviside factor. Default to False
        """
        if gmm.n_components > 1:
            if heaviside is False:
                heaviside = 1

            colors_ = list(six.iteritems(colors.cnames))

            if isinstance(maxi, int) or isinstance(maxi, float64) or isinstance(maxi, float):
                dim = 1
            else:
                dim = len(maxi)
            weight = gmm.weights_
            mean = gmm.means_
            cov = gmm.covars_
            gauss = cls.create_multivar_normal_dis(
                dim, mean, heaviside * cov)
            prob, sort = cls.proba_cluster(
                dim, weight, gauss, x_)
            if dim == 1:
                fig = plt.figure()
                x = np.linspace(mini, maxi)
                for i in range(len(weight)):
                    plt.plot(x_, prob[:, i], ls='--')
                plt.xlabel('Input Values')
                plt.ylabel('Membership probabilities')
                plt.title('Cluster Map')

                fig = plt.figure()
                for i in range(len(sort)):
                    color_ind = int(
                        ((len(colors_) - 1) / sort.max()) * sort[i])
                    color = colors_[color_ind][0]
                    plt.plot(x_[i], y_[i], c=color, marker='o')
                plt.xlabel('Input Values')
                plt.ylabel('Output Values')
                plt.title('Samples with clusters')

            if dim == 2:
                x = fct.map_2d_space(maxi, mini, num=20)
                prob = cls.proba_cluster(
                    dim, weight, gauss, x)[0]

                fig = plt.figure()
                ax1 = fig.add_subplot(111, projection='3d')
                for i in range(len(weight)):
                    color = colors_[
                        int(((len(colors_) - 1) / len(weight)) * i)][0]
                    ax1.plot_trisurf(x[:, 0], x[:, 1], prob[:, i], alpha=0.4,
                                     linewidth=0, color=color)
                plt.title('Cluster Map 3D')

                # fig1 = plt.figure()
                for i in range(len(weight)):
                    color = colors_[
                        int(((len(colors_) - 1) / len(weight)) * i)][0]
                    plt.tricontour(x[:, 0], x[:, 1], prob[:, i], 1, colors=color,
                                   linewidths=3)
                plt.title('Cluster Map 2D')

                fig = plt.figure()
                ax2 = fig.add_subplot(111, projection='3d')
                for i in range(len(sort)):
                    color = colors_[
                        int(((len(colors_) - 1) / sort.max()) * sort[i])][0]
                    ax2.scatter(x_[i][0], x_[i][1], y_[i], c=color)
                plt.title('Samples with clusters')
            plt.show()

#######################################################################

    def _plot_number_cluster(self, posi, errori, erroris, median_eh, median_es, b_ic, a_ic, error_h, error_s, cluster):  # pragma: no cover
        """
        Plot errors (smooth/hard) in function of the number of clusters
        Parameters:
        -----------
        - posi:
        - errori:
        - erroris:
        - median_eh:
        - median_es:
        - b_ic:
        - a_ic:
        - error_h:
        - errors_s
        - cluster:
        """
        ei = []
        eis = []
        meh = []
        mes = []
        b = []
        a = []
        eh = []
        es = []
        pos = []
        for i in posi:
            ei.append(errori[i])
            eis.append(erroris[i])
            meh.append(median_eh[i])
            mes.append(median_es[i])
            b.append(b_ic[i])
            a.append(a_ic[i])
            eh.append(error_h[i])
            es.append(error_s[i])
            pos.append(i + 1)
        # Cross validation
        fig = plt.figure()
        plt.subplot(121)
        predict_hard, = plt.plot(pos, ei, 'bo', linestyle='-')
        predict_smooth, = plt.plot(pos, eis, 'ro', linestyle='-')
        for j in posi:
            possible = plt.axvspan(j + 0.5, j + 1.5, alpha=0.2)
        choix = plt.axvspan(cluster - 0.5, cluster +
                            0.5, alpha=0.2, color='g')
        plt.legend([predict_hard, predict_smooth, possible, choix], ['Hard', 'Smooth',
                                                                     'Possible values',
                                                                     'Best Number of cluster'])
        plt.ylabel('Error MSE')
        plt.xlabel('Number of Cluster')
        plt.title('Evolution of cross-validation error - MSE')
        plt.subplot(122)
        predict_hard, = plt.plot(pos, meh, 'bo', linestyle='-')
        predict_smooth, = plt.plot(pos, mes, 'ro', linestyle='-')
        for j in posi:
            possible = plt.axvspan(j + 0.5, j + 1.5, alpha=0.2)
        choix = plt.axvspan(cluster - 0.5, cluster +
                            0.5, alpha=0.2, color='g')
        plt.legend([predict_hard, predict_smooth, possible, choix], ['Hard', 'Smooth',
                                                                     'Possible values',
                                                                     'Best Number of cluster'])
        plt.ylabel('Medians Of errors')
        plt.xlabel('Number of Cluster')
        plt.title('Evolution of cross-validation error - Median')

        # Box plot BIC
        fig1 = plt.figure()
        plt.subplot(221)
        box_smooth = plt.boxplot(b, positions=pos)
        for j in posi:
            possible = plt.axvspan(j + 0.5, j + 1.5, alpha=0.2)
        choix = plt.axvspan(cluster - 0.5, cluster +
                            0.5, alpha=0.2, color='g')
        plt.legend(
            [possible, choix], ['Possible values', 'Best Number of cluster'])
        plt.title('Box plot for discontinuous recombinaison (BIC)')
        plt.xlabel('Cluster')
        plt.ylabel('')

        # Box plot AIC
        plt.subplot(222)
        box_smooth = plt.boxplot(a, positions=pos)
        for j in posi:
            possible = plt.axvspan(j + 0.5, j + 1.5, alpha=0.2)
        choix = plt.axvspan(cluster - 0.5, cluster +
                            0.5, alpha=0.2, color='g')
        plt.legend(
            [possible, choix], ['Possible values', 'Best Number of cluster'])
        plt.title('Box plot for smooth recombinaison (AIC)')
        plt.xlabel('Cluster')
        plt.ylabel('')

        # Box plot Error hard
        plt.subplot(223)
        box_smooth = plt.boxplot(eh, positions=pos)
        for j in posi:
            possible = plt.axvspan(j + 0.5, j + 1.5, alpha=0.2)
        choix = plt.axvspan(cluster - 0.5, cluster +
                            0.5, alpha=0.2, color='g')
        plt.title('Box plot for discontinuous recombinaison (CV)')
        plt.legend(
            [possible, choix], ['Possible values', 'Best Number of cluster'])
        plt.xlabel('Cluster')
        plt.ylabel('Error')

        # Box plot Error smooth
        plt.subplot(224)
        box_smooth = plt.boxplot(es, positions=pos)
        for j in posi:
            possible = plt.axvspan(j + 0.5, j + 1.5, alpha=0.2)
        choix = plt.axvspan(cluster - 0.5, cluster +
                            0.5, alpha=0.2, color='g')
        plt.title('Box plot for smooth recombinaison (CV)')
        plt.legend(
            [possible, choix], ['Possible values', 'Best Number of cluster'])
        plt.xlabel('Cluster')
        plt.ylabel('Error')
        plt.show()

##########################################################################

    def save(self, file_name):
        """
        This function saves a mixture in a file whose name is file_name
        Parameters:
        ----------
        - mixture: MoE object
        the mixture to save
        - file_name: str
        the name of the file where the mixture is saved
        """
        for index, model in enumerate(self.model_list_name):
            if model == 'RadBF':
                self.model_list[index].model._function = None
                self.model_list[index].model.norm = None
        f = open(file_name, 'wb')
        pickle.dump(self, f)
        f.close()

#######################################################################

    @staticmethod
    def load(file_name):
        """
        This function loads a mixture from a file whose name is file_name
        Parameters:
        ----------
        - file_name: str
        the name of the file where the mixture is loaded
        Return:
        ----------
        - mixture: MoE object
        the mixture loaded
        """
        f = open(file_name, 'rb')
        mixture = pickle.load(f)
        f.close()
        for index, model in enumerate(mixture.model_list_name):
            if model == 'RadBF':
                x = np.array([[1, 2], [3, 4]])
                y = np.array([5, 6])
                intermediate_model = Rbf(x[:, 0], x[:, 1], y)
                mixture.model_list[index].model._function = intermediate_model._function
                mixture.model_list[index].model.norm = intermediate_model.norm
        return mixture

#######################################################################

    @staticmethod
    def format_time(t_sec):  # pragma: no cover
        """
        Format t_sec, ie t_sec = 3661 -> 1h:01m:01s
        t_sec: explicit
        returns: string
        """
        m, s = divmod(t_sec, 60)
        h, m = divmod(m, 60)
        return "%dh:%02dm:%02ds" % (h, m, s)

#######################################################################

    @staticmethod
    def check_input_data(dimension, x, y, c):
        """
        Check that the data given by a user is correct
        Parameters:
        -----------
        - dimension: int
        dimension of the problem
        - x: array_like
        input points
        - y: array_like
        output points
        - c: array_like
        cluster criterion weights
        """
        if x.ndim == 1:
            if dimension != 1:
                raise ValueError("The dimension of the problem %d doesn t match with x dimension 1."
                                 % (dimension))
        else:
            if dimension != len(x[0]):
                raise ValueError("The dimension of the problem %d doesn t match with x dimension %d."
                                 % (dimension, len(x[0])))
        if x.shape[0] != len(y):
            raise ValueError("The number of input points %d doesn t match with the number of output points %d."
                             % (x.shape[0], len(y)))
        if len(y) != len(c):
            raise ValueError("The number of output points %d doesn t match with the number of criterion weights %d."
                             % (len(y), len(c)))
