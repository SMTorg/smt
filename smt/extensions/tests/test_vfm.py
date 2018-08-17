'''
Author: Mohamed Amine Bouhlel <mbouhlel@umich.edu>

This package is distributed under New BSD license.
'''

import unittest
import matplotlib
matplotlib.use('Agg')


class TestVFM(unittest.TestCase):
    @staticmethod
    def run_vfm_example(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import linalg
        from smt.utils import compute_rms_error

        from smt.problems import WaterFlowLFidelity, WaterFlow
        from smt.sampling_methods import LHS
        from smt.extensions import VFM

        # Problem set up
        ndim = 8
        ntest = 500
        ncomp = 1
        ndoeLF = int(10*ndim)
        ndoeHF = int(3)
        funLF = WaterFlowLFidelity(ndim=ndim)
        funHF = WaterFlow(ndim=ndim)
        deriv1 = True
        deriv2 = True
        LF_candidate = 'QP'
        Bridge_candidate = 'KRG'
        type_bridge = 'Multiplicative'
        optionsLF = {}
        optionsB = {'theta0':[1e-2]*ndim,'print_prediction': False,'deriv':False}

        # Construct low/high fidelity data and validation points
        sampling = LHS(xlimits=funLF.xlimits,criterion='m')
        xLF = sampling(ndoeLF)
        yLF = funLF(xLF)
        if deriv1:
            dy_LF = np.zeros((ndoeLF,1))
            for i in range(ndim):
                yd = funLF(xLF,kx=i)
                dy_LF = np.concatenate((dy_LF,yd),axis=1)

        sampling = LHS(xlimits=funHF.xlimits,criterion ='m')
        xHF = sampling(ndoeHF)
        yHF = funHF(xHF)
        if deriv2:
            dy_HF = np.zeros((ndoeHF,1))
            for i in range(ndim):
                yd = funHF(xHF,kx=i)
                dy_HF = np.concatenate((dy_HF,yd),axis=1)

        xtest = sampling(ntest)
        ytest = funHF(xtest)
        dytest = np.zeros((ntest,ndim))
        for i in range(ndim):
            dytest[:,i] = funHF(xtest,kx=i).T

        # Initialize the extension VFM
        M = VFM(type_bridge=type_bridge,name_model_LF=LF_candidate,name_model_bridge=
                Bridge_candidate,X_LF=xLF,y_LF=yLF,X_HF=xHF,y_HF=yHF,options_LF=
                optionsLF, options_bridge = optionsB, dy_LF = dy_LF, dy_HF = dy_HF)

        # Appliy the VFM algorithm
        M.apply_method()

        # Prediction of the validation points
        y = M.analyse_results(x=xtest,operation = 'predict_values')

        plt.figure()
        plt.plot(ytest,ytest,'-.')
        plt.plot(ytest,y,'.')
        plt.xlabel(r'$y$ True')
        plt.ylabel(r'$y$ prediction')
        plt.show()

if __name__ == '__main__':
    unittest.main()
    
