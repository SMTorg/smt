"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

Metamodels - a base class for metamodel methods
"""
#TODO: Extend to multifidelity problems by adding training_pts = {'approx': {}}

from __future__ import division

import numpy as np
from smt.utils import Printer, Timer


class SM(object):
    '''
    Base class for all model methods.
    '''

    def __init__(self, sm_options=None, printf_options=None):
        '''
        Constructor.

        Arguments
        ---------
        sm_options : dict
            Model-related options, in _default_options in the inheriting class

        printf_options : dict
            Output printing options
        '''
        #Initialization
        self._set_default_options()
        self.sm_options.update(sm_options)
        self.printf_options.update(printf_options)

        self.training_pts = {'exact': {}}

        self.printer = Printer()
        self.timer = Timer()

    #############################################################################
    # Model functions
    #############################################################################

    def add_training_pts(self, typ, xt, yt, kx=None):
        '''
        Adds nt training/sample data points

        Arguments
        ---------
        typ : str
            'exact'  if this data are considered as a high-fidelty data
            'approx' if this data are considered as a low-fidelity data (TODO)
        xt : np.ndarray [nt, dim]
            Training point input variable values
        yt : np.ndarray [nt, 1]
            Training point output variable values or derivatives (a vector)
        kx : int, optional
            None if this data set represents output variable values
            int  if this data set represents derivatives
                 where it is differentiated w.r.t. the kx^{th}
                 input variable (kx is 0-based)
        '''
        yt = yt.reshape((xt.shape[0],1))
        #Output or derivative variables
        if kx is None:
            kx = 0
            self.dim = xt.shape[1]
            self.nt = xt.shape[0]
        else:
            kx = kx + 1

        #Construct the input data
        pts = self.training_pts[typ]
        if kx in pts:
            pts[kx][0] = np.vstack([pts[kx][0], xt])
            pts[kx][1] = np.vstack([pts[kx][1], yt])
        else:
            pts[kx] = [np.array(xt), np.array(yt)]

    def train(self):
        '''
        Train the model
        '''
        n_exact = self.training_pts['exact'][0][0].shape[0]

        self.printer.active = self.printf_options['global']
        self.printer._line_break()
        self.printer._center(self.sm_options['name'])

        self.printer.active = self.printf_options['global'] and self.printf_options['problem']
        self.printer._title('Problem size')
        self.printer('   %-25s : %i' % ('# training pts.', n_exact))
        self.printer()

        self.printer.active = self.printf_options['global'] and self.printf_options['time_train']
        if self.sm_options['name'] == 'MixExp':
            # Mixture of experts model
            self.printer._title('Training of the Mixture of experts')
        else:
            self.printer._title('Training')

        #Train the model using the specified model-method
        self.timer._start('fit')
        self.fit()
        self.timer._stop('fit')

        self.printer()
        self.printer._total_time('Total training time (sec)', self.timer['fit'])
        self.printer()

    def predict(self, x):
        '''
        Evaluates the model at a set of unknown points

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

        self.printer.active = self.printf_options['global'] and self.printf_options['time_eval']

        if self.sm_options['name'] == 'MixExp':
            # Mixture of experts model
            self.printer._title('Evaluation of the Mixture of experts')
        else:
            self.printer._title('Evaluation')
        self.printer('   %-12s : %i' % ('# eval pts.', n_evals))

        #Evaluate the unknown points using the specified model-method
        self.timer._start('predict')
        y = self.evaluate(x)
        self.timer._stop('predict')

        self.printer()
        self.printer._total_time('Total prediction time (sec)', self.timer['predict'])
        self.printer._total_time('Time/pt. (sec)', self.timer['predict'] / n_evals)
        self.printer()

        return y.reshape(n_evals,1)
