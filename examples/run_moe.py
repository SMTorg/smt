"""
Test all the functions
"""
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

    #@unittest.skip('disabled')
    def test_norm1_d2_200(self):
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
        mix = MOE(hard_recombination=True, number_cluster=3)
        mix.options['xt'] = xt
        mix.options['yt'] = yt
        mix.apply_method()

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

    #@unittest.skip('disabled')
    def test_branin_d2_100(self):
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
        mix = MOE(hard_recombination=True, number_cluster=6)
        mix.options['xt'] = xt
        mix.options['yt'] = yt
        mix.apply_method()

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
    TestMOE.plot = '--plot' in argv
    unittest.main()
