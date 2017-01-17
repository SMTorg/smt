"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>
        Dr. John T. Hwang         <hwangjt@umich.edu>

Metamodels - a base class for metamodel methods
"""
#TODO: Extend to multifidelity problems by adding training_pts = {'approx': {}}

from __future__ import division

import numpy as np
import time

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
        self.times = {}
        self.printstr_time = '   %-14s : %10.7f'


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

        if self.printf_options['global']:
            self._print_line_break()
            self._print_line(self.sm_options['name'], True)

        if self.printf_options['global'] and self.printf_options['problem']:
            self._print_problem()

        #Train the model using the specified model-method
        t1 = time.time()
        self.fit()
        t2 = time.time()

        # Mixture of experts model
        if self.printf_options['global'] and self.printf_options['time_train']:
            if self.sm_options['name'] == 'MixExp':
                self._print_line_title('Training of the Mixture of experts')
            else:
                self._print_line_title('Training')
            print
            self._print_line_time('Total (sec)', t2-t1)
            print


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

        #Initialization
        print_eval = self.printf_options['global'] and \
                     self.printf_options['time_eval']
        n_evals = x.shape[0]

        # If mixture of experts model
        if print_eval:
            if self.sm_options['name'] == 'MixExp':
                self._print_line_title('Evaluation of the Mixture of experts')
            else:
                self._print_line_title('Evaluation')
            string = '   %-12s : %i'
            print string % ('# eval pts.', n_evals)

        #Evaluate the unknown points using the specified model-method
        t1 = time.time()
        y = self.evaluate(x)
        t2 = time.time()

        self.times['time/pt'] = (t2-t1)/n_evals
        if print_eval:
            print
            self._print_line_time('Total (sec)', t2-t1)
            self._print_line_time('Time/pt. (sec)', (t2-t1)/n_evals)
            print

        return y.reshape(n_evals,1)


    #############################################################################
    # Print functions
    #############################################################################


    def _print_line_break(self):

        print '_' * 75
        print


    def _print_line(self, string, center=False):

        if center:
            pre = ' ' * int((75 - len(string))/2.0)
        else:
            pre = ''
        print pre + '%s' % string


    def _print_line_time(self, name, time, string=''):

        print '   %-14s : %10.7f' % (name, time), string


    def _print_line_title(self, title):

        self._print_line_break()
        self._print_line(' ' + title)
        print


    def _print_problem(self):

        pts = self.training_pts
        self._print_line_title('Problem size')
        nexact = self.training_pts['exact'][0][0].shape[0]

        string = '   %-25s : %i'
        print string % ('# training pts. (exact)', nexact)
        print
