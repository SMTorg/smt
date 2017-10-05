"""
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.

Mixture of Experts
"""

from __future__ import division
import numpy as np

from smt.utils.options_dictionary import OptionsDictionary
from smt.extensions2.extensions import Extensions

from smt.mixture.cluster import create_clustering, sort_values_by_cluster, create_multivar_normal_dis, proba_cluster
from smt.mixture.utils import sum_x, cut_list, concat_except
from smt.mixture.error import Error
#from smt.mixture.factory import ModelFactory
from smt.utils.misc import compute_rms_error
from statsmodels.tools.eval_measures import rmse

class MOE(Extensions):
    
    def _initialize(self):
        super(MOE, self)._initialize()
        declare = self.options.declare
        
        declare('X', None, types=np.ndarray, desc='Training inputs')
        declare('y', None, types=np.ndarray, desc='Training outputs')
        declare('c', None, types=np.ndarray, desc='Clustering training outputs')
        declare('number_cluster', 0, types=int, desc='Number of cluster')
        declare('hard_recombination', True, types=bool, desc='Steep cluster transition')

    def _apply(self):
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
        if self.options['X'] is not None and self.options['y'] is not None:
            x = self.options['X']
            y = self.options['y']
            c = self.options['c']
            number_cluster = self.options['number_cluster']
        else:
            raise ValueError('Check X, y')
        
        if c is None:
            c = y

        self._check_input_data(x, y, c)

        self.x = x
        self.y = y
        self.c = c
        self.scale_factor=1.
        self.hard = self.options['hard_recombination']
        dimension = x.shape[1]
        reset = True
        median = True
        valid = None
        max_number_cluster = None
        heaviside = False

        if reset:
            self.model_list = []
            self.model_list_name = []
            self.model_list_param = []

        # choice of test values and trained values
        self.values = np.c_[x, y, c]

        if valid is None:
            cut_values = cut_list(self.values, 10)  # cut space in 10 parts
            training_values = np.vstack(concat_except(cut_values, 0))  # assemble 9 parts
            test_values = np.vstack(cut_values[0])
        else:
            training_values = self.values
            test_values = valid

        self.test_values = test_values
        self.training_values = training_values

        total_length = training_values.shape[1]
        x_trained = training_values[:, 0:dimension]
        y_trained = training_values[:, dimension]
        c_trained = training_values[:, dimension + 1:total_length]

        # choice of number of cluster
        if reset:
            if max_number_cluster is None:
                max_number_cluster = int(len(x) / 10) + 1

            if number_cluster is 0:
                self._default_number_cluster(dimension, x, y, c,
                                             median=median, max_number_cluster=max_number_cluster)
            else:
                self.number_cluster = number_cluster

            if self.number_cluster > max_number_cluster:
                print 'Number of clusters should be inferior to {0}'.format(max_number_cluster)
                raise ValueError(
                    'The number of clusters is too high considering the number of points')

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

        # Choice of the models and training
        self._fit_without_clustering(
            dimension, x_trained, y_trained, c_trained, new_model=reset)

        # choice of heaviside factor
        if heaviside and self.number_cluster > 1:
            self._best_scale_factor(
                test_values[:, 0:dimension], test_values[:, dimension], detail=detail, plot=plot)

        self.gauss = create_multivar_normal_dis(self.dimension, self.cluster.means_,
                                                    self.scale_factor * self.cluster.covars_)

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
            self._fit_without_clustering(dimension, x, y, c, new_model=False)

        #gc.collect()
    
    def _analyse_results(self,x,operation = 'predict_values',kx=None):
        pass 
    

    def _check_input_data(self, x, y, c):
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
        if x.shape[0] != y.shape[0]:
            raise ValueError("The number of input points %d doesn t match with the number of output points %d."
                             % (x.shape[0], y.shape[0]))
        if y.shape[0] != c.shape[0]:
            raise ValueError("The number of output points %d doesn t match with the number of criterion weights %d."
                             % (y.shape[0], c.shape[0]))


    def _default_number_cluster(self, dim, x, y, c, available_models=None, max_number_cluster=None, median=True):
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
                    #available_models = [{'type': 'RadBF'}]
                    available_models = [{'type': 'QP', 'degree': 2}]
                else:
                    available_models = [{'type': 'QP', 'degree': 2}]

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
                    mixture = MOE()
                    mixture._surrogate_type = {'QP': mixture._surrogate_type['QP']}
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
                            mixture._fit_without_clustering(dim, val_train[:, 0:dim], val_train[:, dim],
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

#             if detail:  # pragma: no cover
#                 # Print details about the clustering
#                 print 'Number of cluster:', i + 1, '| Possible:', ok, '| Error(hard):', np.mean(errorc), '| Error(smooth):', np.mean(errors), '| Median (hard):', np.median(errorc), '| Median (smooth):', np.median(errors)
#                 print '#######'

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

        if median:
            method = '| Method: Minimum of Median errors'
        else:
            method = '| Method: Minimum of relative L2'

    def _fit_without_clustering(self, dimension, x_trained, y_trained, c_trained, new_model=True):
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
        self._check_input_data(x_trained, y_trained, c_trained)

        self.dimension = dimension

        self.gauss = create_multivar_normal_dis(self.dimension, self.cluster.means_,
                                                self.scale_factor * self.cluster.covars_)

        sort_cluster = self.cluster.predict(np.c_[x_trained, c_trained])

        # sort trained_values for each cluster
        trained_cluster = sort_values_by_cluster(
            np.c_[x_trained, y_trained], self.number_cluster, sort_cluster)

        # find model for each cluster
        for clus in range(self.number_cluster):

            if new_model:

                if len(self._surrogate_type) == 1:
                    x_trained = np.array(trained_cluster[clus])[:, 0:dimension]
                    y_trained = np.array(trained_cluster[clus])[:, dimension]
                    model = self.use_model()
                    model.set_training_values(x_trained, y_trained)
                    model.train()
                else:
                    model = self._find_best_model(trained_cluster[clus])

                self.model_list.append(model)
                #self.model_list_name.append(model.name)
                #self.model_list_param.append(param)

            else:  # Train on the overall domain
                trained_values = np.array(trained_cluster[clus])
                x_trained = trained_values[:, 0:dimension]
                y_trained = trained_values[:, dimension]
                self.model_list[clus].set_training_values(x_trained, y_trained)
                self.model_list[clus].train()

        #gc.collect()

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

#         for model in self.factory.available_models:
# 
#             if model['type'] == 'PA':
#                 min_number_point = np.math.factorial(self.dimension + model['degree']) / (
#                     np.math.factorial(self.dimension) * np.math.factorial(model['degree']))
#                 ok = self._check_number_points(
#                     trained_cluster, min_number_point)
# 
#             if model['type'] == 'PA2' or model['type'] == 'PA2smt':
#                 min_number_point = np.math.factorial(self.dimension + 2) / (
#                     np.math.factorial(self.dimension) * np.math.factorial(2))
#                 ok = self._check_number_points(
#                     trained_cluster, min_number_point)
# 
#             if model['type'] == 'LS':
#                 min_number_point = 1 + self.dimension
#                 ok = self._check_number_points(
#                     trained_cluster, min_number_point)
# 
#             if model['type'] in ['Krig', 'KrigPLS', 'KrigPLSK']:
# 
#                 if model['regr'] == 'linear':
#                     min_number_point = 1 + self.dimension
#                     ok = self._check_number_points(
#                         trained_cluster, min_number_point)
# 
#                 if model['regr'] == 'quadratic':
#                     min_number_point = np.math.factorial(self.dimension + 2) / (
#                         np.math.factorial(self.dimension) * np.math.factorial(2))
#                     ok = self._check_number_points(
#                         trained_cluster, min_number_point)
# 
#             if ok is False:
#                 return False

        return True
    
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
        dimension = self.dimension
        sorted_trained_values = np.array(sorted_trained_values)
        
        rmses = {}
        sms = {}

        for name, sm_class in self._surrogate_type.iteritems():
            print name
            if name == 'RMTC' or name == 'RMTB' or name == 'GEKPLS' or name == 'KRG':
                continue
            
            sm = sm_class()
            sm.set_training_values(sorted_trained_values[:, 0:dimension], sorted_trained_values[:, dimension])
            sm.train()
            
            rmses[sm.name] = compute_rms_error(sm)
            sms[sm.name] = sm
            
        best_name=None
        best_rmse=None
        for name, rmse in rmses.iteritems():
            if best_rmse is None or rmse < best_rmse:
                best_name, best_rmse = name, rmse              
            
        return sms[best_name]

        # Training for each model
#         for i, params in enumerate(self.factory.available_models):
# #             try:
#             model, param = self.factory.create_trained_model(
#                 sorted_trained_values[:, 0:dimension], sorted_trained_values[:, dimension], dimension, params)  # Training
#             erreurm = error.Error(self.test_values[:, dimension], model.predict_output(
#                 self.test_values[:, 0:dimension]))  # errors
#             model_list.append(model)
#             param_model_list.append(param)
#             error_list_by_model.append(erreurm.l_two_rel)
#             name_model_list.append(
#                 self.factory.available_models[i]['type'])

#             except NameError:
#                 if detail:  # pragma: no cover
#                     print self.factory.available_models[i]['type'] + " can not be performed"

        # find minimum error
#        min_error_index = error_list_by_model.index(min(error_list_by_model))

#         # find best model and train it with all the training samples
#         model = model_list[min_error_index]
#         param = param_model_list[min_error_index]
#         model_name = name_model_list[min_error_index]
# 
#         return model, param, model_name 

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
        self.valid_hard = Error(np.atleast_2d(y).T, yh)
        self.valid_smooth = Error(np.atleast_2d(y).T, ys)
        
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
            predicted_values.append(model.predict_values(np.atleast_2d(x[i]))[0])
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
                    self.model_list[j].predict_values(np.atleast_2d(x[i]))[0] * sort_proba[i][j]

            predicted_values.append(recombined_value)

        predicted_values = np.array(predicted_values)

        return predicted_values