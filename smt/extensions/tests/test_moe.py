'''
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.
'''

import unittest
import numpy as np
from sys import argv

from smt.extensions import MOE
from smt.utils.sm_test_case import SMTestCase
from smt.problems import Branin, LpNorm
from smt.sampling_methods import FullFactorial
from smt.utils.misc import compute_rms_error

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TestMOE(SMTestCase):
    """
    Test class
    """
    plot = None

    @staticmethod
    def function_test_1d(x):
        x = np.reshape(x, (-1, ))
        y = np.zeros(x.shape)
        y[x<0.4] = x[x<0.4]**2
        y[(x>=0.4) & (x<0.8)] = 3*x[(x>=0.4) & (x<0.8)]+1  
        y[x>=0.8] = np.sin(10*x[x>=0.8])
        return y.reshape((-1, 1))

    #@unittest.skip('disabled')
    def test_1d_50(self):
        self.ndim = 1
        self.nt = 50
        self.ne = 50 

        np.random.seed(0)
        xt = np.random.sample(self.nt).reshape((-1, 1))
        yt = self.function_test_1d(xt)
        moe = MOE(smooth_recombination=True, 
                  heaviside_optimization=True, 
                  n_clusters=3,
                  xt=xt, yt=yt)   
        moe.train()

        # validation data
        np.random.seed(1)
        xe = np.random.sample(self.ne)
        ye = self.function_test_1d(xe)

        rms_error = compute_rms_error(moe, xe, ye)
        self.assert_error(rms_error, 0., 3e-1)
        if TestMOE.plot:
            y = moe.predict_values(xe)
            plt.figure(1)
            plt.plot(ye, ye,'-.')
            plt.plot(ye, y, '.')
            plt.xlabel(r'$y$ actual')
            plt.ylabel(r'$y$ prediction')
            plt.figure(2)
            xv = np.linspace(0, 1, 100)
            yv = self.function_test_1d(xv)
            plt.plot(xv, yv, '-.')
            plt.plot(xe, y, 'o')
            plt.show()

    #@unittest.skip('disabled')
    def test_norm1_2d_200(self):
        self.ndim = 2
        self.nt = 200
        self.ne = 200

        prob = LpNorm(ndim=self.ndim)

        # training data
        sampling = FullFactorial(xlimits=prob.xlimits, clip=True)
        np.random.seed(0)
        xt = sampling(self.nt)
        yt = prob(xt)

        # mixture of experts
        moe = MOE(smooth_recombination=False, n_clusters=5, xt=xt, yt=yt)     
        moe.train()

        # validation data
        np.random.seed(1)
        xe = sampling(self.ne)
        ye = prob(xe)

        rms_error = compute_rms_error(moe, xe, ye)
        self.assert_error(rms_error, 0., 1e-1)

        if TestMOE.plot:
            y = moe.predict_values(xe)
            plt.figure(1)
            plt.plot(ye, ye,'-.')
            plt.plot(ye, y, '.')
            plt.xlabel(r'$y$ actual')
            plt.ylabel(r'$y$ prediction')

            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xt[:,0], xt[:,1], yt)
            plt.title('L1 Norm')
            plt.show()

    #@unittest.skip('disabled')
    def test_branin_2d_200(self):
        self.ndim = 2
        self.nt = 200
        self.ne = 200

        prob = Branin(ndim=self.ndim)

        # training data
        sampling = FullFactorial(xlimits=prob.xlimits, clip=True)
        np.random.seed(0)
        xt = sampling(self.nt)
        yt = prob(xt)

        # mixture of experts
        moe = MOE(n_clusters=6)
        moe.options['xt'] = xt
        moe.options['yt'] = yt
        moe.options['heaviside_optimization'] = True    
        moe.train()

        # validation data
        np.random.seed(1)
        xe = sampling(self.ne)
        ye = prob(xe)

        rms_error = compute_rms_error(moe, xe, ye)
        self.assert_error(rms_error, 0., 1e-1)

        if TestMOE.plot:
            y = moe.analyse_results(x=xe, operation='predict_values')
            plt.figure(1)
            plt.plot(ye, ye,'-.')
            plt.plot(ye, y, '.')
            plt.xlabel(r'$y$ actual')
            plt.ylabel(r'$y$ prediction')

            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xt[:,0], xt[:,1], yt)
            plt.title('Branin function')
            plt.show()

		
if __name__ == '__main__':
    if '--plot' in argv:
        TestMOE.plot = True
        argv.remove('--plot')
    unittest.main()
