"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

Metamodels - a base class for metamodel methods
"""
#TODO: Extend to multifidelity problems by adding training_points = {'approx': {}}
#TODO: Complete the mixture of expert model: verify from if self.options['name'] == 'MixExp': (predict)

from __future__ import division

import numpy as np

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

        self.training_points = {'exact': {}}

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
            if kx2 not in self.training_points['exact']:
                raise ValueError('There is no training point data available for kx %s' % kx2)
            xt, yt = self.training_points['exact'][kx2]
            if kx == None:
                yt2 = self.predict_value(xt)
            else:
                yt2 = self.predict_derivative(xt, kx)
            num += np.linalg.norm(yt2 - yt) ** 2
            den += np.linalg.norm(yt) ** 2
            return num ** 0.5 / den ** 0.5

    def add_training_points_values(self, typ, xt, yt):
        '''
        Adds nt training/sample data points

        Arguments
        ---------
        typ : str
            'exact'  if this data are considered as a high-fidelty data
            'approx' if this data are considered as a low-fidelity data (TODO)
        xt : np.ndarray [nt, nx]
            Training point input variable values
        yt : np.ndarray [nt, ny]
            Training point output variable values (a vector)
        '''
        nt = xt.shape[0]
        nx = xt.shape[1]
        ny = int(np.prod(yt.shape) / nt)
        yt = yt.reshape((nt, ny))

        self.nx = nx
        self.ny = ny

        kx = 0
        self.dim = xt.shape[1]
        self.nt = xt.shape[0]
        
        #Construct the input data
        pts = self.training_points[typ]
        if kx in pts:
            pts[kx][0] = np.vstack([pts[kx][0], xt])
            pts[kx][1] = np.vstack([pts[kx][1], yt])
        else:
            pts[kx] = [np.array(xt), np.array(yt)]

    def add_training_points_derivatives(self, typ, xt, yt, kx):
        '''
        Adds nt training/sample data points

        Arguments
        ---------
        typ : str
            'exact'  if this data are considered as a high-fidelty data
            'approx' if this data are considered as a low-fidelity data (TODO)
        xt : np.ndarray [nt, nx]
            Training point input variable values
        yt : np.ndarray [nt, ny]
            Training derivatives (a vector)
        kx : int 
            The kx^{th} input variable (kx is 0-based)
        '''
        nt = xt.shape[0]
        nx = xt.shape[1]
        ny = int(np.prod(yt.shape) / nt)
        yt = yt.reshape((nt, ny))

        self.nx = nx
        self.ny = ny

        #Derivative variables
        kx = kx + 1

        #Construct the input data
        pts = self.training_points[typ]
        if kx in pts:
            pts[kx][0] = np.vstack([pts[kx][0], xt])
            pts[kx][1] = np.vstack([pts[kx][1], yt])
        else:
            pts[kx] = [np.array(xt), np.array(yt)]
            
            
    def train(self):
        '''
        Train the model
        '''
        n_exact = self.training_points['exact'][0][0].shape[0]

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
