"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

Metamodels - a base class for metamodel methods
"""
#TODO: Extend to multifidelity problems by adding training_pts = {'approx': {}}

from __future__ import division

import numpy as np
from smt.utils import Timer


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

        self.print_status = True
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
        self._print_line_break()
        self._print(self.sm_options['name'], True)

        self.print_status = self.printf_options['global'] and self.printf_options['problem']
        self.timer.print_status = self.print_status
        self._print_problem()

        self.print_status = self.printf_options['global'] and self.printf_options['time_train']
        self.timer.print_status = self.print_status
        if self.sm_options['name'] == 'MixExp':
            # Mixture of experts model
            self._print_title('Training of the Mixture of experts')
        else:
            self._print_title('Training')

        #Train the model using the specified model-method
        self.timer._start('fit')
        self.fit()
        self.timer._stop('fit')

        self._print()
        self.timer._print('fit', 'Total training time (sec)')
        self._print()

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

        self.print_status = self.printf_options['global'] and self.printf_options['time_eval']
        self.timer.print_status = self.print_status

        # If mixture of experts model
        if self.sm_options['name'] == 'MixExp':
            self._print_title('Evaluation of the Mixture of experts')
        else:
            self._print_title('Evaluation')
        self._print('   %-12s : %i' % ('# eval pts.', n_evals))

        #Evaluate the unknown points using the specified model-method
        self.timer._start('predict')
        y = self.evaluate(x)
        self.timer._stop('predict')

        self._print()
        self.timer._print('predict', 'Total prediction time (sec)')
        self.timer._print('predict', 'Time/pt. (sec)', n_evals)
        self._print()

        return y.reshape(n_evals,1)

    #############################################################################
    # Print functions
    #############################################################################

    def _print_line_break(self):
        self._print('_' * 75)
        self._print()

    def _print(self, string='', center=False):
        if self.print_status:
            if center:
                pre = ' ' * int((75 - len(string))/2.0)
            else:
                pre = ''
            print(pre + '%s' % string)

    def _print_title(self, title):
        self._print_line_break()
        self._print(' ' + title)
        self._print()

    def _print_problem(self):
        pts = self.training_pts
        self._print_title('Problem size')
        nexact = self.training_pts['exact'][0][0].shape[0]
        self._print('   %-25s : %i' % ('# training pts.', nexact))
        self._print()
