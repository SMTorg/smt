'''
Author: Dr. John T. Hwang <hwangjt@umich.edu>
        Dr. Mohamed A. Bouhlel <mbouhlel@umich>

This package is distributed under New BSD license.
'''

from __future__ import print_function, division
import numpy as np
import unittest
import inspect

from six import iteritems
from collections import OrderedDict

from smt.problems import TensorProduct
from smt.sampling_methods import LHS, FullFactorial

from smt.utils.sm_test_case import SMTestCase
from smt.utils.silence import Silence
from smt.utils import compute_rms_error
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS
from smt.extensions import MFK
from copy import deepcopy
try:
    from smt.surrogate_models import IDW, RBF, RMTC, RMTB
    compiled_available = True
except:
    compiled_available = False


print_output = False

class Test(SMTestCase):

    def setUp(self):
        ndim = 3
        nt = 100
        ne = 100
        ncomp = 1

        problems = OrderedDict()
        problems['exp'] = TensorProduct(ndim=ndim, func='exp')
        problems['tanh'] = TensorProduct(ndim=ndim, func='tanh')
        problems['cos'] = TensorProduct(ndim=ndim, func='cos')

        sms = OrderedDict()
        sms['LS'] = LS()
        sms['QP'] = QP()
        sms['KRG'] = KRG(theta0=[1e-2]*ndim)
        sms['MFK'] = MFK(theta0=[1e-2]*ndim)
        sms['KPLS'] = KPLS(theta0=[1e-2]*ncomp,n_comp=ncomp)
        sms['KPLSK'] = KPLSK(theta0=[1]*ncomp,n_comp=ncomp)
        sms['GEKPLS'] = GEKPLS(theta0=[1e-2]*ncomp,n_comp=ncomp,delta_x=1e-1)
        if compiled_available:
            sms['IDW'] = IDW()
            sms['RBF'] = RBF()
            sms['RMTC'] = RMTC()
            sms['RMTB'] = RMTB()

        t_errors = {}
        t_errors['LS'] = 1.0
        t_errors['QP'] = 1.0
        t_errors['KRG'] = 1e-5
        t_errors['MFK'] = 1e-5
        t_errors['KPLS'] = 1e-5
        t_errors['KPLSK'] = 1e-5
        t_errors['GEKPLS'] = 1e-5
        if compiled_available:
            t_errors['IDW'] = 1e-15
            t_errors['RBF'] = 1e-2
            t_errors['RMTC'] = 1e-1
            t_errors['RMTB'] = 1e-1

        e_errors = {}
        e_errors['LS'] = 1.5
        e_errors['QP'] = 1.5
        e_errors['KRG'] = 1e-2
        e_errors['MFK'] = 1e-2
        e_errors['KPLS'] = 1e-2
        e_errors['KPLSK'] = 1e-2
        e_errors['GEKPLS'] = 1e-2
        if compiled_available:
            e_errors['IDW'] = 1e0
            e_errors['RBF'] = 1e0
            e_errors['RMTC'] = 2e-1
            e_errors['RMTB'] = 2e-1

        self.nt = nt
        self.ne = ne
        self.ndim = ndim
        self.problems = problems
        self.sms = sms
        self.t_errors = t_errors
        self.e_errors = e_errors

    def run_test(self):
        method_name = inspect.stack()[1][3]
        pname = method_name.split('_')[1]
        sname = method_name.split('_')[2]

        prob = self.problems[pname]
        sampling = FullFactorial(xlimits=prob.xlimits, clip=True)

        np.random.seed(0)
        xt = sampling(self.nt)
        yt = prob(xt)
        print(prob(xt,kx=0).shape)
        for i in range(self.ndim):
            yt = np.concatenate((yt,prob(xt,kx=i)),axis=1)

        np.random.seed(1)
        xe = sampling(self.ne)
        ye = prob(xe)

        sm0 = self.sms[sname]

        sm = sm0.__class__()
        sm.options = sm0.options.clone()
        if sm.options.is_declared('xlimits'):
            sm.options['xlimits'] = prob.xlimits
        sm.options['print_global'] = False

        sm.set_training_values(xt, yt[:, 0])
        if sm.supports['training_derivatives']:
            for i in range(self.ndim):
                sm.set_training_derivatives(xt,yt[:, i+1],i)

        with Silence():
            sm.train()

        t_error = compute_rms_error(sm)
        e_error = compute_rms_error(sm, xe, ye)
        
    def run_MF_test(self):
        method_name = inspect.stack()[1][3]
        pname = method_name.split('_')[1]
        sname = method_name.split('_')[2]

        prob = self.problems[pname]
        sampling = FullFactorial(xlimits=prob.xlimits, clip=True)

        np.random.seed(0)
        xt = sampling(self.nt)
        yt = prob(xt)
        print(prob(xt,kx=0).shape)
        for i in range(self.ndim):
            yt = np.concatenate((yt,prob(xt,kx=i)),axis=1)
        
        y_lf = 2*prob(xt) + 2
        x_lf = deepcopy(xt) 
        np.random.seed(1)
        xe = sampling(self.ne)
        ye = prob(xe)

        sm0 = self.sms[sname]

        sm = sm0.__class__()
        sm.options = sm0.options.clone()
        if sm.options.is_declared('xlimits'):
            sm.options['xlimits'] = prob.xlimits
        sm.options['print_global'] = False

        sm.set_training_values(xt, yt[:, 0])
        sm.set_training_values(x_lf, y_lf[:, 0], name = 0)
        if sm.supports['training_derivatives']:
            for i in range(self.ndim):
                sm.set_training_derivatives(xt,yt[:, i+1],i)

        with Silence():
            sm.train()

        t_error = compute_rms_error(sm)
        e_error = compute_rms_error(sm, xe, ye)
    # --------------------------------------------------------------------
    # Function: exp

    def test_exp_LS(self):
        self.run_test()

    def test_exp_QP(self):
        self.run_test()

    def test_exp_KRG(self):
        self.run_test()

    def test_exp_KPLS(self):
        self.run_test()

    def test_exp_KPLSK(self):
        self.run_test()

    def test_exp_GEKPLS(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_exp_IDW(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_exp_RBF(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_exp_RMTC(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_exp_RMTB(self):
        self.run_test()
    
    def test_exp_MFK(self):
        self.run_MF_test()
        
    # --------------------------------------------------------------------
    # Function: tanh

    def test_tanh_LS(self):
        self.run_test()

    def test_tanh_QP(self):
        self.run_test()

    def test_tanh_KRG(self):
        self.run_test()

    def test_tanh_KPLS(self):
        self.run_test()

    def test_tanh_KPLSK(self):
        self.run_test()

    def test_tanh_GEKPLS(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_tanh_IDW(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_tanh_RBF(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_tanh_RMTC(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_tanh_RMTB(self):
        self.run_test()
    
    def test_tanh_MFK(self):
        self.run_MF_test()
    # --------------------------------------------------------------------
    # Function: cos

    def test_cos_LS(self):
        self.run_test()

    def test_cos_QP(self):
        self.run_test()

    def test_cos_KRG(self):
        self.run_test()

    def test_cos_KPLS(self):
        self.run_test()

    def test_cos_KPLSK(self):
        self.run_test()

    def test_cos_GEKPLS(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_cos_IDW(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_cos_RBF(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_cos_RMTC(self):
        self.run_test()

    @unittest.skipIf(not compiled_available, 'Compiled Fortran libraries not available')
    def test_cos_RMTB(self):
        self.run_test()
    
    def test_cos_MFK(self):
        self.run_MF_test()
    

if __name__ == '__main__':
    print_output = True
    print('%6s %8s %18s %18s'
          % ('SM', 'Problem', 'Train. pt. error', 'Test pt. error'))
    unittest.main()
