KPLS
====

KPLS is a kriging model that uses the partial least squares (PLS) method. KPLS is faster than kriging because of the low number of hyperparameters to be estimated while maintaining a good accuracy.
This model is suitable for high-dimensional problems due to the kernel constructed through the PLS method. The PLS method [1]_ is a well known tool for high-dimensional problems that searches the direction that maximizes the variance between the input and output variables. This is done by a projection in a smaller space spanned by the so-called principal components. The PLS information are integrated into the kriging correlation matrix to scale the number of inputs by reducing the number of hyperparameters. The number of principal components $h$, which corresponds to the number of hyperparameters for KPLS and much lower than :math:`nx` (number of dimension of the problem), usually does not exceed 4 components:

.. math ::
  \prod\limits_{l=1}^{nx}\exp\left(-\theta_l\left(x_l^{(i)}-x_l^{(j)}\right)^2\right),\qquad \qquad \qquad\prod\limits_{l=1}^{nx} \prod\limits_{k=1}^h\exp\left(-\theta_k\left(w_l^{(k)}x_l^{(i)}-w_l^{(k)}x_l^{(j)}\right)^{2}\right) \quad \forall\ \theta_l,\theta_h\in\mathbb{R}^+\\
  \text{Standard Gaussian correlation function} \quad \qquad\text{PLS-Gaussian correlation function}\qquad \qquad\qquad\quad

More details about the KPLS approach could be found in these sources [2]_.

.. [1] Wold, H., Soft modeling by latent variables: the nonlinear iterative partial least squares approach, Perspectives in probability and statistics, papers in honour of MS Bartlett, 1975, pp. 520--540.

.. [2] Bouhlel, M. A. and Bartoli, N. and  Otsmane, A. and Morlier, J., Improving kriging surrogates of high-dimensional design models by Partial Least Squares dimension reduction, Structural and Multidisciplinary Optimization, Vol. 53, No. 5, 2016, pp. 935--952.

Usage
-----

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.surrogate_models import KPLS
  
  xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
  yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])
  
  sm = KPLS(theta0=[1e-2])
  sm.set_training_values(xt, yt)
  sm.train()
  
  num = 100
  x = np.linspace(0.0, 4.0, num)
  y = sm.predict_values(x)
  
  plt.plot(xt, yt, "o")
  plt.plot(x, y)
  plt.xlabel("x")
  plt.ylabel("y")
  plt.legend(["Training data", "Prediction"])
  plt.show()
  
::

  ___________________________________________________________________________
     
                                     KPLS
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 5
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
     Training - done. Time (sec):  0.0000000
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: kpls_Test_test_kpls.png
  :scale: 80 %
  :align: center

Options
-------

.. list-table:: List of options
  :header-rows: 1
  :widths: 15, 10, 20, 20, 30
  :stub-columns: 0

  *  -  Option
     -  Default
     -  Acceptable values
     -  Acceptable types
     -  Description
  *  -  print_global
     -  True
     -  None
     -  ['bool']
     -  Global print toggle. If False, all printing is suppressed
  *  -  print_training
     -  True
     -  None
     -  ['bool']
     -  Whether to print training information
  *  -  print_prediction
     -  True
     -  None
     -  ['bool']
     -  Whether to print prediction information
  *  -  print_problem
     -  True
     -  None
     -  ['bool']
     -  Whether to print problem information
  *  -  print_solver
     -  True
     -  None
     -  ['bool']
     -  Whether to print solver information
  *  -  poly
     -  constant
     -  ['constant', 'linear', 'quadratic']
     -  None
     -  Regression function type
  *  -  corr
     -  squar_exp
     -  ['abs_exp', 'squar_exp']
     -  None
     -  Correlation function type
  *  -  data_dir
     -  None
     -  None
     -  ['str']
     -  Directory for loading / saving cached data; None means do not save or load
  *  -  theta0
     -  [0.01]
     -  None
     -  ['list', 'ndarray']
     -  Initial hyperparameters
  *  -  n_comp
     -  1
     -  None
     -  ['int']
     -  Number of principal components
