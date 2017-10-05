"""
This file is an interface for moe.
Thanks to this factory, trained model can be created in moe_main.py
"""

import numpy as np
from numpy import double

#from moe.models.kriging import Kriging
#from moe.models.ls import LeastSquare
#from moe.models.pa2 import PA2
#from moe.models.pa import PA
#from moe.models.rbf import RadBF
from smt.methods.smt_adaptor import SMTModelAdaptor

#try:
#    import kpls.kpls as kpls
#    from moe.models.kriging_partial_least_square import KrigPLS
#    from moe.models.kplsk import KrigPLSK
#    FLAG_KPLS = True
#except ImportError:
#    FLAG_KPLS = False
FLAG_KPLS = False

# try:
#     import mfk.mf_gaussian_process as COK
#     from moe.models.muti_fidelity_kriging import MFKrig
#     FLAG_COK = True
# except ImportError:
#     FLAG_COK = False
FLAG_COK = False

try:
    from smt.methods.ls import LS
#     from smt.methods.pa2 import PA2 as PA2_smt
#     from smt.methods.kpls import KPLS
    FLAG_SMT = True
except ImportError:
    FLAG_SMT = False

MACHINE_EPSILON = np.finfo(double).eps


class UnknownModelException(RuntimeError):
    """
    A class for model type error
    """
    pass


class UninstalledModelException(RuntimeError):
    """
    A class for model type error
    """
    pass


class ModelFactory(object):
    """
    A class which deals with the available models and
    allows to make trained models.
    Attributes :
    -----------
    - available_models :
    A list of dictionaries which are the different possible models and its parameters
    """

    def __init__(self, optim=False, sigma=False):
        """
        Initialize the ModelFactory object
        Optional:
        ----------
        - optim: boolean
        Set to True to have models with jacobian predictions
        - sigma: Boolean
        Set to True to have models with sigmas and jacobian predictions
        """

        self.available_models = self._initialize_models_list(optim, sigma)

    def _get_available_models(self):
        """
        This method allows to get the different possible models
        """
        return self.available_models

    def set_models(self, available_models):
        """
        This method allows to set the different models and their parameters
        Parameters :
        -----------
        - available_models :
        A list of dictionaries which are the different possible models and its parameters
        """

        self.available_models = []

        for i in range(len(available_models)):
            model = available_models[i]['type']
            params = available_models[i]

            if model in ['KrigPLS', 'KrigPLSK']:
                if FLAG_KPLS:
                    self.available_models.append({})
                    self.available_models[i] = self._initialize_parameters(
                        model)

                    for j in range(len(params)):
                        param = params.keys()[j]
                        value = params.values()[j]
                        self.available_models[i][param] = value
                else:
                    print(
                        'You want to use KPLS model whereas you have not installed kpls module')
                    raise UninstalledModelException()

            elif model in ['LS_smt', 'PA2_smt', 'KRG_smt']:
                if FLAG_SMT:
                    self.available_models.append({})
                    self.available_models[i] = self._initialize_parameters(
                        model)

                    for j in range(len(params)):
                        param = params.keys()[j]
                        value = params.values()[j]
                        self.available_models[i][param] = value
                else:
                    print(
                        'You want to use models from SMT whereas you have not installed SMT')
                    raise UninstalledModelException()

            else:
                self.available_models.append({})
                self.available_models[i] = self._initialize_parameters(model)

                for j in range(len(params)):
                    param = params.keys()[j]
                    value = params.values()[j]
                    self.available_models[i][param] = value

    def add_model(self, model):
        """
        This method allows to add a model and its parameters to the available models list
        Parameters :
        -----------
        - model :
        A dictionary represents a model and its parameters
        """
        model_name = model['type']
        params = model

        n = len(self.available_models)

        if model_name in ['KrigPLS', 'KrigPLSK']:
            if FLAG_KPLS:
                self.available_models.append({})

                self.available_models[n] = self._initialize_parameters(
                    model_name)

                for j in range(len(params)):
                    param = params.keys()[j]
                    value = params.values()[j]
                    self.available_models[n][param] = value
            else:
                print(
                    'You want to use KPLS model whereas you have not installed kpls module')
                raise UninstalledModelException()

        elif model_name in ['LS_smt', 'PA2_smt', 'KRG_smt']:
            if FLAG_SMT:
                self.available_models.append({})
                self.available_models[n] = self._initialize_parameters(
                    model_name)

                for j in range(len(params)):
                    param = params.keys()[j]
                    value = params.values()[j]
                    self.available_models[n][param] = value
            else:
                print(
                    'You want to use models from SMT whereas you have not installed SMT')
                raise UninstalledModelException()

        else:
            self.available_models.append({})
            self.available_models[n] = self._initialize_parameters(model_name)

            for j in range(len(params)):
                param = params.keys()[j]
                value = params.values()[j]
                self.available_models[n][param] = value

    def _initialize_models_list(self, optim, sigma):
        """
        This method allows to initialize the different possible models
        """
        available_models = []
        regr_lists_ok = ['constant', 'linear', 'quadratic']
        regr_lists_kpls = ['constant', 'linear']
        corr_lists = ['absolute_exponential',
                      'squared_exponential', 'matern32', 'matern52']

        if sigma:
            if FLAG_KPLS:
                available_models.append(self._initialize_parameters(
                    'KrigPLSK', corr='squared_exponential', regr='constant'))
                available_models.append(self._initialize_parameters(
                    'KrigPLSK', corr='squared_exponential', regr='linear'))
            for corr in corr_lists:
                for regr in regr_lists_ok:
                    available_models.append(self._initialize_parameters(
                        'Krig', corr=corr, regr=regr))
            for corr in corr_lists:
                for regr in regr_lists_kpls:
                    if FLAG_KPLS:
                        available_models.append(self._initialize_parameters(
                            'KrigPLS', corr=corr, regr=regr))

        elif optim:
            available_models.append(self._initialize_parameters('LS'))
            available_models.append(self._initialize_parameters('PA2'))
            if FLAG_KPLS:
                available_models.append(self._initialize_parameters(
                    'KrigPLSK', corr='squared_exponential', regr='constant'))
                available_models.append(self._initialize_parameters(
                    'KrigPLSK', corr='squared_exponential', regr='linear'))
            for corr in corr_lists:
                for regr in regr_lists_ok:
                    available_models.append(
                        self._initialize_parameters('Krig', corr=corr, regr=regr))
            for corr in corr_lists:
                for regr in regr_lists_kpls:
                    if FLAG_KPLS:
                        available_models.append(self._initialize_parameters(
                            'KrigPLS', corr=corr, regr=regr))

        else:
#             available_models.append(self._initialize_parameters('LS'))
#             available_models.append(self._initialize_parameters('PA2'))
#             available_models.append(self._initialize_parameters('PA'))
#             available_models.append(self._initialize_parameters('RadBF'))
# 
#             if FLAG_KPLS:
#                 available_models.append(self._initialize_parameters(
#                     'KrigPLSK', corr='squared_exponential', regr='constant'))
#                 available_models.append(self._initialize_parameters(
#                     'KrigPLSK', corr='squared_exponential', regr='linear'))
#             for corr in corr_lists:
#                 for regr in regr_lists_ok:
#                     available_models.append(
#                         self._initialize_parameters('Krig', corr=corr, regr=regr))
#             for corr in corr_lists:
#                 for regr in regr_lists_kpls:
#                     if FLAG_KPLS:
#                         available_models.append(self._initialize_parameters(
#                             'KrigPLS', corr=corr, regr=regr))

            if FLAG_SMT:
#                available_models.append(self._initialize_parameters('PA2smt'))
#                available_models.append(self._initialize_parameters('KRGsmt'))
                available_models.append(self._initialize_parameters('LSsmt'))

        return available_models

    @staticmethod
    def _initialize_parameters(name, corr='squared_exponential', regr='constant'):
        """
        This method allows to initialize by default the parameters of a model
        Parameters :
        -----------
        - name : str
        name of the model
        Optional :
        -----------
        - corr : str
        type of the correlation kernel
        - regr ; str
        type of the regression kernel
        """
        params = {}

        if name == 'Krig':
            params['type'] = 'Krig'
            params['theta0'] = [1]
            params['thetaU'] = None
            params['thetaL'] = None
            params['regr'] = regr
            params['corr'] = corr
            params['normalize'] = True
            params['beta0'] = None
            params['storage_mode'] = 'full'
            params['verbose'] = False
            params['optimizer'] = 'fmin_cobyla'
            params['random_start'] = 1
            params['nugget'] = 10. * MACHINE_EPSILON
            params['random_state'] = None

        elif name == 'MFK':
            params['type'] = 'MFK'
            params['verbose'] = False
            params['optimizer'] = 'fmin_cobyla'
            params['normalize'] = False
            params['corr'] = corr
            params['beta0'] = None
            params['regr'] = regr
            params['rho_regr'] = 'constant'
            params['theta0'] = [5]
            params['thetaL'] = [0.1]
            params['thetaU'] = [10]
            params['noise_estim'] = False
            params['noise0'] = 1e-3
            params['nugget'] = 10. * MACHINE_EPSILON
            params['random_start'] = 1
            params['random_state'] = None

        elif name == 'KrigPLS':
            params['type'] = 'KrigPLS'
            params['n_components'] = 1
            params['theta0'] = [1]
            params['thetaL'] = [0.1]
            params['thetaU'] = [10]
            params['rhobeg'] = 1
            params['rhoend'] = 1e-3
            params['regr'] = regr
            params['corr'] = corr
            params['optimizer'] = 'fmin_cobyla'
            params['type_pls'] = 'CoefPLS_Regression'
            params['max_iter'] = int(1e5)
            params['tol'] = 1e-06
            params['nb_ill_matrix'] = 5
            params['storage_mode'] = 'full'
            params['maxfun'] = 1e6
            params['normalize'] = True
            params['nugget'] = 10. * MACHINE_EPSILON

        elif name == 'KrigPLSK':
            params['type'] = 'KrigPLSK'
            params['n_components'] = 1
            params['theta0'] = [1]
            params['thetaL'] = [0.1]
            params['thetaU'] = [10]
            params['rhobeg'] = 1
            params['rhoend'] = 1e-3
            params['regr'] = regr
            params['corr'] = corr
            params['optimizer'] = 'fmin_cobyla'
            params['type_pls'] = 'CoefPLS_Regression'
            params['max_iter'] = int(1e5)
            params['tol'] = 1e-06
            params['nb_ill_matrix'] = 5
            params['storage_mode'] = 'full'
            params['maxfun'] = 1e6
            params['normalize'] = True
            params['nugget'] = 10. * MACHINE_EPSILON

        elif name == 'LS':
            params['type'] = 'LS'
            params['fit_intercept'] = True
            params['normalize'] = False
            params['copy_X'] = True
            params['n_jobs'] = 1

        elif name == 'PA':
            params['type'] = 'PA'
            params['degree'] = 3
            params['interaction_only'] = False
            params['include_bias'] = True
            params['alpha'] = 1.0
            params['fit_intercept'] = True
            params['normalize'] = False
            params['copy_X'] = True
            params['max_iter'] = None
            params['tol'] = 1e-3
            params['solver'] = "auto"
            params['random_state'] = None

        elif name == 'PA2':
            params['type'] = 'PA2'

        elif name == 'RadBF':
            params['type'] = 'RadBF'

        elif name == 'LSsmt':
            params['type'] = 'LSsmt'

        elif name == 'PA2smt':
            params['type'] = 'PA2smt'

        elif name == 'KRGsmt':
            params['type'] = 'KRGsmt'
            params['name'] = 'KPLS'
            params['n_comp'] = 1
            params['theta0'] = [1e-2]
            params['delta_x'] = 1e-4
            params['poly'] = 'constant'
            params['corr'] = 'squar_exp'
            params['best_iteration_fail'] = None
            params['nb_ill_matrix'] = 5

        else:
            print "The name {0} is not correct for a type of model".format(name)
            raise UnknownModelException()

        return params

    def create_trained_model(self, x, y, dim, params):
        """
        This function allows to create trained model
        Parameters :
        -----------
        - x : Array_like
        x sample
        - y : Array_like
        y sample
        - dim : int
        dimension of the problem
        - params
        dictionary of the parameters for one model
        """
        name = params['type']

        if name == 'Krig':

            model = Kriging(regr=params['regr'], corr=params['corr'], beta0=params['beta0'],
                            storage_mode=params['storage_mode'], verbose=params['verbose'],
                            theta0=params['theta0'], thetaL=params['thetaL'], thetaU=params[
                'thetaU'], optimizer=params['optimizer'], random_start=params[
                'random_start'], normalize=params['normalize'],
                nugget=params['nugget'], random_state=params['random_state'])
            model.check_params(params)

            return self._train_model(model, x, y, params)

        elif name == 'MFK':

            model = MFKrig(verbose=params['verbose'], optimizer=params['optimizer'], normalize=params[
                'normalize'], corr=params['corr'], beta0=params['beta0'], regr=params['regr'],
                rho_regr=params['rho_regr'], theta0=params['theta0'], thetaL=params[
                'thetaL'], thetaU=params['thetaU'], noise_estim=params['noise_estim'],
                noise0=params['noise0'], nugget=params['nugget'], random_start=params[
                'random_start'], random_state=params['random_state'])
            # model.check_params(params)

            return self._train_model(model, x, y, params)

        elif name == 'KrigPLS':

            params['n_components'] = self._set_n_components(params, dim)

            params = self._reshape_thetas(params, params['n_components'])

            model = KrigPLS(n_components=params['n_components'], theta0=params['theta0'], thetaL=params[
                'thetaL'], thetaU=params['thetaU'], rhobeg=params['rhobeg'], rhoend=params['rhoend'],
                regr=params['regr'], corr=params['corr'], optimizer=params['optimizer'],
                type_pls=params['type_pls'], max_iter=params['max_iter'], tol=params[
                'tol'], nb_ill_matrix=params['nb_ill_matrix'], storage_mode=params[
                'storage_mode'], maxfun=params['maxfun'], normalize=params['normalize'])
            model.check_params(params)

            return self._train_model(model, x, y, params)

        elif name == 'KrigPLSK':

            params['n_components'] = self._set_n_components(params, dim)

            params = self._reshape_thetas(params, params['n_components'])

            model = KrigPLSK(n_components=params['n_components'], theta0=params['theta0'], thetaL=params['thetaL'],
                             thetaU=params['thetaU'], rhobeg=params['rhobeg'], rhoend=params['rhoend'],
                             regr=params['regr'], corr=params['corr'], optimizer=params['optimizer'],
                             type_pls=params['type_pls'], max_iter=params['max_iter'], tol=params['tol'],
                             nb_ill_matrix=params['nb_ill_matrix'], storage_mode=params['storage_mode'],
                             maxfun=params['maxfun'], normalize=params['normalize'])
            model.check_params(params)

            return self._train_model(model, x, y, params)

        elif name == 'LS':

            model = LeastSquare(fit_intercept=params['fit_intercept'], normalize=params['normalize'],
                                copy_X=params['copy_X'], n_jobs=params['n_jobs'])
            model.check_params(params)

            return self._train_model(model, x, y, params)

        elif name == 'LSsmt':

            smt_model = LS()
            model = SMTModelAdaptor(smt_model)

            return self._train_model(model, x, y, params)

        elif name == 'PA2smt':
            smt_model = PA2_smt()
            model = SMTModelAdaptor(smt_model)

            return self._train_model(model, x, y, params)

        elif name == 'KRGsmt':

            smt_model = KPLS(name=params['name'], n_comp=params['n_comp'], theta0=params['theta0'],
                             delta_x=params['delta_x'], poly=params['poly'], corr=params['corr'],
                             best_iteration_fail=params['best_iteration_fail'], nb_ill_matrix=params['nb_ill_matrix'])
            model = SMTModelAdaptor(smt_model)

            return self._train_model(model, x, y, params)

        elif name == 'PA2':

            model = PA2()

            return self._train_model(model, x, y, params)

        elif name == 'PA':

            model = PA(degree=params['degree'], interaction_only=params['interaction_only'],
                       include_bias=params['include_bias'], alpha=params['alpha'],
                       fit_intercept=params['fit_intercept'], normalize=params['normalize'],
                       copy_X=params['copy_X'], max_iter=params['max_iter'], tol=params['tol'],
                       solver=params['solver'], random_state=params['random_state'])
            model.check_params(params)

            return self._train_model(model, x, y, params)

        elif name == 'RadBF':

            model = RadBF()

            return self._train_model(model, x, y, params)

        else:
            print "The name {0} is not correct for a type of model".format(name)
            raise UnknownModelException()

    @staticmethod
    def _train_model(model, x, y, params):
        """
        This function trains the model once created
        Parameters :
        -----------
        - model :
        Object of model type
        - x : Array_like
        x sample
        - y : Array_like
        y sample
        - params :
        dictionary of the parameters for one model
        """
        if params['type'] == 'KrigPLSK':
            model.train_kpls(x, y, params)

        model.train(x, y)

        if params['type'] in ['Krig', 'MFK', 'KrigPLS']:
            if model.model.thetaU is not None:
                model, params = model.improve_theta_up(x, y, params)

        return model, params

    @staticmethod
    def _reshape_thetas(params, dim):
        """
        This function reshapes the thetas if needed
        Parameters :
        -----------
        - params :
        dictionary of the parameters for one model
        - dim : int
        dimension of the problem
        """

        if np.array(params['theta0']).shape[0] != dim:
            params['theta0'] = np.resize(params['theta0'], dim)

        if params['thetaU'] is not None:
            if np.array(params['thetaU']).shape[0] != dim:
                params['thetaU'] = np.resize(params['thetaU'], dim)

        if params['thetaL'] is not None:
            if np.array(params['thetaL']).shape[0] != dim:
                params['thetaL'] = np.resize(params['thetaL'], dim)

        return params

    @staticmethod
    def _set_n_components(params, dim):
        """
        This function set n_components
        Parameters :
        -----------
        - params :
        dictionary of the parameters for one model
        - dim : int
        dimension of the problem
        """

        n_components = params['n_components']
        if dim < n_components:
            n_components = dim
        if n_components is None:
            if dim > 5:
                n_components = 3
            else:
                n_components = dim

        return n_components
