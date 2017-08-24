'''
Author: Dr. John T. Hwang <hwangjt@umich.edu>
        Dr. Mohamed A. Bouhlel <mbouhlel@umich.edu>
        
This package is distributed under New BSD license.
'''

import unittest

import matplotlib
matplotlib.use('Agg')

try:
    from smt.methods import IDW, RBF, RMTB, RMTC
    compiled_available = True
except:
    compiled_available = False


class Test(unittest.TestCase):

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_idw(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import IDW

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        sm = IDW(p=2)
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rbf(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import RBF

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        sm = RBF(d0=5)
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtb(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import RMTB

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        xlimits = np.array([[0., 4.]])

        sm = RMTB(xlimits=xlimits, order=4, num_ctrl_pts=20, reg_dv=1e-15, reg_cons=1e-15)
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtc(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import RMTC

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        xlimits = np.array([[0., 4.]])

        sm = RMTC(xlimits=xlimits, num_elements=20, reg_dv=1e-15, reg_cons=1e-15)
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    def test_ls(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import LS

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        sm = LS()
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    def test_qp(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import QP

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        sm = QP()
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    def test_krg(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import KRG

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        sm = KRG(theta0=[1e-2])
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    def test_kpls(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import KPLS

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        sm = KPLS(theta0=[1e-2])
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)

        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    def test_kplsk(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from smt.methods import KPLSK

        xt = np.array([0., 1., 2., 3., 4.])
        yt = np.array([0., 1., 1.5, 0.5, 1.0])

        sm = KPLSK(theta0=[1e-2])
        sm.set_training_values(xt, yt)
        sm.train()

        num = 100
        x = np.linspace(0., 4., num)
        y = sm.predict_values(x)
        yy = sm.predict_derivatives(xt,0)        
        plt.plot(xt, yt, 'o')
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Training data', 'Prediction'])
        plt.show()

    def test_gekpls(self):
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        from smt.methods import GEKPLS
        from smt.problems import Sphere
        from smt.sampling import LHS

        # Construction of the DOE
        fun = Sphere(ndim = 2)
        sampling = LHS(xlimits=fun.xlimits,criterion = 'm')
        xt = sampling(20)
        yt = fun(xt)
        # Compute the gradient
        for i in range(2):
            yd = fun(xt,kx=i)
            yt = np.concatenate((yt,yd),axis=1)

        # Build the GEKPLS model
        sm = GEKPLS(theta0=[1e-2],xlimits=fun.xlimits,extra_points=1,print_prediction = False)
        sm.set_training_values(xt, yt[:,0])
        for i in range(2):
            sm.set_training_derivatives(xt,yt[:, 1+i].reshape((yt.shape[0],1)),i)
        sm.train()

        # Test the model
        X = np.arange(fun.xlimits[0,0], fun.xlimits[0,1], .25)
        Y = np.arange(fun.xlimits[1,0], fun.xlimits[1,1], .25)
        X, Y = np.meshgrid(X, Y)        
        Z = np.zeros((X.shape[0],X.shape[1]))

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = sm.predict_values(np.hstack((X[i,j],Y[i,j])).reshape((1,2)))
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z)
 
        plt.show()

if __name__ == '__main__':
    unittest.main()
