"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

Metamodels - a base class for metamodel methods
"""
#TODO: Extend to multifidelity problems by adding training_points = {'approx': {}}
#TODO: Complete the mixture of expert model: verify from if self.options['name'] == 'MixExp': (predict)

from __future__ import division

import numpy as np
from collections import defaultdict

from smt.utils.printer import Printer
from smt.utils.options_dictionary import OptionsDictionary


class SM(object):
    '''
    Base class for all model methods.
    '''

    def __init__(self, **kwargs):
        '''
        Constructor.

        Arguments
        ---------
        **kwargs : named arguments
            Set of options that can be optionally set; each option must have been declared.
        '''
        self.options = OptionsDictionary()
        self._declare_options()
        self.options.update(kwargs)

        self.training_points = defaultdict(dict)

        self.printer = Printer()


    def _declare_options(self):
        declare = self.options.declare

        declare('print_global', True, types=bool,
                desc='Global print toggle. If False, all printing is suppressed')
        declare('print_training', True, types=bool,
                desc='Whether to print training information')
        declare('print_prediction', True, types=bool,
                desc='Whether to print prediction information')
        declare('print_problem', True, types=bool,
                desc='Whether to print problem information')
        declare('print_solver', True, types=bool,
                desc='Whether to print solver information')

    def compute_rms_error(self, xe=None, ye=None, kx=None):
        '''
        Returns the RMS error of the training points or the given points.

        Arguments
        ---------
        xe : np.ndarray[ne, dim] or None
            Input values. If None, the input values at the training points are used instead.
        ye : np.ndarray[ne, 1] or None
            Output / deriv. values. If None, the training pt. outputs / derivs. are used.
        kx : int or None
            If None, we are checking the output values.
            If int, we are checking the derivs. w.r.t. the kx^{th} input variable (0-based).
        '''
        if xe is not None and ye is not None:
            if kx == None:
                ye2 = self.predict_value(xe)
            else:
                ye2 = self.predict_derivative(xe, kx)
            return np.linalg.norm(ye2 - ye) / np.linalg.norm(ye)
        elif xe is None and ye is None:
            num = 0.
            den = 0.
            if kx is None:
                kx2 = 0
            else:
                kx2 += 1
            if kx2 not in self.training_points[None]:
                raise ValueError('There is no training point data available for kx %s' % kx2)
            xt, yt = self.training_points[None][kx2]
            if kx == None:
                yt2 = self.predict_value(xt)
            else:
                yt2 = self.predict_derivative(xt, kx)
            num += np.linalg.norm(yt2 - yt) ** 2
            den += np.linalg.norm(yt) ** 2
            return num ** 0.5 / den ** 0.5

    def set_training_values(self, xt, yt, name=None):
        '''
        Set training data (values).

        Arguments
        ---------
        xt : np.ndarray[nt, nx]
            The input values for the nt training points.
        yt : np.ndarray[nt, ny]
            The output values for the nt training points.
        name : str or None
            An optional label for the group of training points being set.
            This is only used in special situations (e.g., multi-fidelity applications).
        '''
        if not isinstance(xt, np.ndarray):
            raise ValueError('xt must be a NumPy array')
        if not isinstance(yt, np.ndarray):
            raise ValueError('yt must be a NumPy array')

        xt = np.atleast_2d(xt.T).T
        yt = np.atleast_2d(yt.T).T

        if len(xt.shape) != 2:
            raise ValueError('xt must have a rank of 1 or 2')
        if len(yt.shape) != 2:
            raise ValueError('yt must have a rank of 1 or 2')
        if xt.shape[0] != yt.shape[0]:
            raise ValueError('the first dimension of xt and yt must have the same length')

        self.nt = xt.shape[0]
        self.nx = xt.shape[1]
        self.ny = yt.shape[1]

        kx = 0
        self.dim = xt.shape[1]

        self.training_points[name][kx] = [np.array(xt), np.array(yt)]

    def set_training_derivatives(self, xt, dyt_dxt, kx, name=None):
        '''
        Set training data (derivatives).

        Arguments
        ---------
        xt : np.ndarray[nt, nx]
            The input values for the nt training points.
        dyt_dxt : np.ndarray[nt, ny]
            The derivatives values for the nt training points.
        kx : int
            0-based index of the derivatives being set.
        name : str or None
            An optional label for the group of training points being set.
            This is only used in special situations (e.g., multi-fidelity applications).
        '''
        if not isinstance(xt, np.ndarray):
            raise ValueError('xt must be a NumPy array')
        if not isinstance(dyt_dxt, np.ndarray):
            raise ValueError('dyt_dxt must be a NumPy array')
        if not isinstance(kx, int):
            raise ValueError('kx must be an int')

        xt = np.atleast_2d(xt.T).T
        dyt_dxt = np.atleast_2d(dyt_dxt.T).T

        if len(xt.shape) != 2:
            raise ValueError('xt must have a rank of 1 or 2')
        if len(dyt_dxt.shape) != 2:
            raise ValueError('dyt_dxt must have a rank of 1 or 2')
        if xt.shape[0] != dyt_dxt.shape[0]:
            raise ValueError('the first dimension of xt and dyt_dxt must have the same length')

        self.training_points[name][kx + 1] = [np.array(xt), np.array(dyt_dxt)]

    def train(self):
        '''
        Train the model
        '''
        n_exact = self.training_points[None][0][0].shape[0]

        self.printer.active = self.options['print_global']
        self.printer._line_break()
        self.printer._center(self.name)

        self.printer.active = self.options['print_global'] and self.options['print_problem']
        self.printer._title('Problem size')
        self.printer('   %-25s : %i' % ('# training points.', n_exact))
        self.printer()

        self.printer.active = self.options['print_global'] and self.options['print_training']
        if self.name == 'MixExp':
            # Mixture of experts model
            self.printer._title('Training of the Mixture of experts')
        else:
            self.printer._title('Training')

        #Train the model using the specified model-method
        with self.printer._timed_context('Training', 'training'):
            self._train()

    def predict_value(self, x):
        '''
        Evaluates the model at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values

        Returns
        -------
        y : np.ndarray
            Evaluation point output variable values
        '''
        n_evals = x.shape[0]

        self.printer.active = self.options['print_global'] and self.options['print_prediction']

        if self.name == 'MixExp':
            # Mixture of experts model
            self.printer._title('Evaluation of the Mixture of experts')
        else:
            self.printer._title('Evaluation')
        self.printer('   %-12s : %i' % ('# eval points.', n_evals))
        self.printer()

        #Evaluate the unknown points using the specified model-method
        with self.printer._timed_context('Predicting', key='prediction'):
            y = self._predict_value(x)

        time_pt = self.printer._time('prediction')[-1] / n_evals
        self.printer()
        self.printer('Prediction time/pt. (sec) : %10.7f' %  time_pt)
        self.printer()

        return y.reshape(n_evals, self.ny)

    def predict_derivative(self, x, kx):
        '''
        Evaluates the derivatives at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        kx : int
            The 0-based index of the input variable with respect to which derivatives are desired.

        Returns
        -------
        y : np.ndarray
            Derivative values.
        '''
        n_evals = x.shape[0]

        self.printer.active = self.options['print_global'] and self.options['print_prediction']

        if self.name == 'MixExp':
            # Mixture of experts model
            self.printer._title('Evaluation of the Mixture of experts')
        else:
            self.printer._title('Evaluation')
        self.printer('   %-12s : %i' % ('# eval points.', n_evals))
        self.printer()

        #Evaluate the unknown points using the specified model-method
        with self.printer._timed_context('Predicting', key='prediction'):
            y = self._predict_derivative(x, kx)

        time_pt = self.printer._time('prediction')[-1] / n_evals
        self.printer()
        self.printer('Prediction time/pt. (sec) : %10.7f' %  time_pt)
        self.printer()

        return y.reshape(n_evals, self.ny)

    def predict_variance(self, x):
        '''
        Evaluates the variance at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values.

        Returns
        -------
        y : np.ndarray
            Variance values.
        '''
        return self._predict_variance(x)

    def _predict_derivative(self, x, kx):
        '''
        Evaluates the derivatives at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values
        kx : int
            The 0-based index of the input variable with respect to which derivatives are desired.

        Returns
        -------
        y : np.ndarray
            Derivative values.
        '''
        raise NotImplementedError('Derivative evaluation is not implemented for this SM method.')

    def _predict_variance(self, x):
        '''
        Evaluates the variance at a set of points.

        Arguments
        ---------
        x : np.ndarray [n_evals, dim]
            Evaluation point input variable values.

        Returns
        -------
        y : np.ndarray
            Variance values.
        '''
        raise NotImplementedError('Variance prediction is not implemented for this SM method.')
