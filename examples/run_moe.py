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
from smt.sampling import FullFactorial
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
        self.nt = 30
        self.ne = 30 

        np.random.seed(0)
        xt = np.random.sample(self.nt)
        yt = self.function_test_1d(xt)
        mix = MOE(smooth_recombination=True, 
                  heaviside_optimization=False, 
                  n_clusters=3)
        mix.options['xt'] = xt.reshape((-1, 1))
        mix.options['yt'] = yt     
        mix.train()

        # validation data
        np.random.seed(1)
        xe = np.random.sample(self.ne)
        ye = self.function_test_1d(xe)

        rms_error = compute_rms_error(mix, xe, ye)
        #self.assert_error(rms_error, 0., 1e-1)
        if TestMOE.plot:
            y = mix.predict_values(xe)
            # plt.figure(1)
            # plt.plot(xt, yt,'x')
            # plt.plot(xe, y,'o')
            # plt.figure(2)
            # plt.plot(ye, y,'o')
            # from scipy.stats import multivariate_normal
            # plt.figure(1)
            # for g in mix.gauss:
            #     x = np.linspace(0, 1, 100)
            #     y = multivariate_normal.pdf(x, mean=g.mean, cov=g.cov)
            #     plt.plot(x, y)
            # plt.show()

    @unittest.skip('disabled')
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
        mix = MOE(smooth_recombination=False, n_clusters=5)
        mix.options['xt'] = xt
        mix.options['yt'] = yt     
        mix.train()

        # validation data
        np.random.seed(1)
        xe = sampling(self.ne)
        ye = prob(xe)

        rms_error = compute_rms_error(mix, xe, ye)
        self.assert_error(rms_error, 0., 1e-1)

        if TestMOE.plot:
            y = mix.predict_values(xe)
            plt.figure(1)
            plt.plot(ye, ye,'-.')
            plt.plot(ye, y, '.')
            plt.xlabel(r'$y$ actual')
            plt.ylabel(r'$y$ prediction')

            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xt[:,0], xt[:,1], yt)
            plt.show()

    @unittest.skip('disabled')
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
        mix = MOE(n_clusters=6)
        mix.options['xt'] = xt
        mix.options['yt'] = yt
        mix.options['heaviside_optimization'] = True    
        mix.train()

        # validation data
        np.random.seed(1)
        xe = sampling(self.ne)
        ye = prob(xe)

        rms_error = compute_rms_error(mix, xe, ye)
        self.assert_error(rms_error, 0., 1e-1)

        if TestMOE.plot:
            y = mix.analyse_results(x=xe, operation='predict_values')
            plt.figure(1)
            plt.plot(ye, ye,'-.')
            plt.plot(ye, y, '.')
            plt.xlabel(r'$y$ actual')
            plt.ylabel(r'$y$ prediction')

            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xt[:,0], xt[:,1], yt)
            plt.show()

		
if __name__ == '__main__':
    if '--plot' in argv:
        TestMOE.plot = True
        argv.remove('--plot')
    unittest.main()
