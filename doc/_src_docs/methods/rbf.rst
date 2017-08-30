Radial basis functions
======================

The radial basis function (RBF) surrogate model represents the interpolating function
as a linear combination of basis functions, one for each training point.
RBFs are named as such because the basis functions depend only on
the distance from the prediction point to the training point for the basis function.
The coefficients of the basis functions are computed during the training stage.
RBFs are frequently augmented to global polynomials to capture the general trends.

The prediction equation for RBFs is

.. math ::
  y = \mathbf{p}(\mathbf{x}) \mathbf{w_p} + \sum_i^{nt} \mathbf{p}hi(\mathbf{x}, \mathbf{xt}_i) \mathbf{w_r} ,

where
:math:`\mathbf{x} \in \mathbb{R}^{nx}` is the prediction input vector,
:math:`y \in \mathbb{R}` is the prediction output,
:math:`\mathbf{xt}_i \in \mathbb{R}^{nx}` is the input vector for the :math:`i` th training point,
:math:`\mathbf{p}(\mathbf{x}) \in \mathbb{R}^{np}` is the vector mapping the polynomial coefficients to the prediction output,
:math:`\mathbf{p}hi(\mathbf{x}, \mathbf{xt}_i) \in \mathbb{R}^{nt}` is the vector mapping the radial basis function coefficients to the prediction output,
:math:`\mathbf{w_p} \in \mathbb{R}^{np}` is the vector of polynomial coefficients,
and
:math:`\mathbf{w_r} \in \mathbb{R}^{nt}` is the vector of radial basis function coefficients.

The coefficients, :math:`\mathbf{w_p}` and :math:`\mathbf{w_r}`, are computed by solving the follow linear system:

.. math ::

  \mathbf{b}egin{bmatrix}
    \mathbf{p}hi( \mathbf{xt}_1 , \mathbf{xt}_1 ) & \dots & \mathbf{p}hi( \mathbf{xt}_1 , \mathbf{xt}_{nt} ) & \mathbf{p}( \mathbf{xt}_1 ) ^ T \\
    \vdots & \ddots & \vdots & \vdots \\
    \mathbf{p}hi( \mathbf{xt}_{nt} , \mathbf{xt}_1 ) & \dots & \mathbf{p}hi( \mathbf{xt}_{nt} , \mathbf{xt}_{nt} ) & \mathbf{p}( \mathbf{xt}_{nt} ) ^ T \\
    \mathbf{p}( \mathbf{xt}_1 ) & \dots & \mathbf{p}( \mathbf{xt}_{nt} ) & \mathbf{0} \\
  \end{bmatrix}
  \mathbf{b}egin{bmatrix}
    \mathbf{w_r}_1 \\
    \vdots \\
    \mathbf{w_r}_{nt} \\
    \mathbf{w_p} \\
  \end{bmatrix}
  =
  \mathbf{b}egin{bmatrix}
    \mathbf{yt}_1 \\
    \vdots \\
    \mathbf{yt}_{nt} \\
    0 \\
  \end{bmatrix}

At the moment, only Gaussian basis functions are implemented.
These are given by:

.. math ::

  \mathbf{p}hi( \mathbf{x}_i , \mathbf{x}_j ) = \exp \left( \frac{|| \mathbf{x}_i - \mathbf{x}_j ||_2 ^ 2}{d0^2} \right)

Usage
-----

.. code-block:: python

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
  
::

  ___________________________________________________________________________
     
                                      RBF
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 5
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Initializing linear solver ...
           Performing LU fact. (5 x 5 mtx) ...
           Performing LU fact. (5 x 5 mtx) - done. Time (sec):  0.0000582
        Initializing linear solver - done. Time (sec):  0.0000949
        Solving linear system (col. 0) ...
           Back solving (5 x 5 mtx) ...
           Back solving (5 x 5 mtx) - done. Time (sec):  0.0000570
        Solving linear system (col. 0) - done. Time (sec):  0.0000899
     Training - done. Time (sec):  0.0006051
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000439
     
     Prediction time/pt. (sec) :  0.0000004
     
  
.. figure:: rbf.png
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
  *  -  data_dir
     -  None
     -  None
     -  ['str']
     -  Directory for loading / saving cached data; None means do not save or load
  *  -  print_solver
     -  True
     -  None
     -  ['bool']
     -  Whether to print solver information
  *  -  print_problem
     -  True
     -  None
     -  ['bool']
     -  Whether to print problem information
  *  -  print_global
     -  True
     -  None
     -  ['bool']
     -  Global print toggle. If False, all printing is suppressed
  *  -  poly_degree
     -  -1
     -  [-1, 0, 1]
     -  ['int']
     -  -1 means no global polynomial, 0 means constant, 1 means linear trend
  *  -  max_print_depth
     -  5
     -  None
     -  ['int']
     -  Maximum depth (level of nesting) to print operation descriptions and times
  *  -  print_training
     -  True
     -  None
     -  ['bool']
     -  Whether to print training information
  *  -  reg
     -  1e-10
     -  None
     -  ['int', 'float']
     -  Regularization coeff.
  *  -  d0
     -  1.0
     -  None
     -  ['int', 'float', 'list', 'ndarray']
     -  basis function scaling parameter in exp(-d^2 / d0^2)
  *  -  print_prediction
     -  True
     -  None
     -  ['bool']
     -  Whether to print prediction information
