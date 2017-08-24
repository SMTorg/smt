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

  \newcommand\RR{\mathbb{R}}
  \newcommand\y{\mathbf{y}}
  \newcommand\x{\mathbf{x}}
  \newcommand\a{\mathbf{a}}
  \newcommand\b{\mathbf{b}}
  \newcommand\p{\mathbf{p}}
  \newcommand\xt{\mathbf{xt}}
  \newcommand\yt{\mathbf{yt}}
  \newcommand\wp{\mathbf{w_p}}
  \newcommand\wr{\mathbf{w_r}}
  \newcommand\sumt{\sum_i^{nt}}
  y = \p(\x) \wp + \sumt \phi(\x, \xt_i) \wr ,

where
:math:`\x \in \RR^{nx}` is the prediction input vector,
:math:`y \in \RR` is the prediction output,
:math:`\xt_i \in \RR^{nx}` is the input vector for the :math:`i` th training point,
:math:`\p(\x) \in \RR^{np}` is the vector mapping the polynomial coefficients to the prediction output,
:math:`\phi(\x, \xt_i) \in \RR^{nt}` is the vector mapping the radial basis function coefficients to the prediction output,
:math:`\wp \in \RR^{np}` is the vector of polynomial coefficients,
and
:math:`\wr \in \RR^{nt}` is the vector of radial basis function coefficients.

The coefficients, :math:`\wp` and :math:`\wr`, are computed by solving the follow linear system:

.. math ::

  \begin{bmatrix}
    \phi( \xt_1 , \xt_1 ) & \dots & \phi( \xt_1 , \xt_{nt} ) & \p( \xt_1 ) ^ T \\
    \vdots & \ddots & \vdots & \vdots \\
    \phi( \xt_{nt} , \xt_1 ) & \dots & \phi( \xt_{nt} , \xt_{nt} ) & \p( \xt_{nt} ) ^ T \\
    \p( \xt_1 ) & \dots & \p( \xt_{nt} ) & \mathbf{0} \\
  \end{bmatrix}
  \begin{bmatrix}
    \wr_1 \\
    \vdots \\
    \wr_{nt} \\
    \wp \\
  \end{bmatrix}
  =
  \begin{bmatrix}
    \yt_1 \\
    \vdots \\
    \yt_{nt} \\
    0 \\
  \end{bmatrix}

At the moment, only Gaussian basis functions are implemented.
These are given by:

.. math ::

  \phi( \x_i , \x_j ) = \exp \left( \frac{|| \x_i - \x_j ||_2 ^ 2}{d0^2} \right)

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
           Performing LU fact. (5 x 5 mtx) - done. Time (sec):  0.0000961
        Initializing linear solver - done. Time (sec):  0.0001287
        Solving linear system (col. 0) ...
           Back solving (5 x 5 mtx) ...
           Back solving (5 x 5 mtx) - done. Time (sec):  0.0000751
        Solving linear system (col. 0) - done. Time (sec):  0.0001009
     Training - done. Time (sec):  0.0006139
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000362
     
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
  *  -  d0
     -  1.0
     -  None
     -  ['int', 'float', 'list', 'ndarray']
     -  basis function scaling parameter in exp(-d^2 / d0^2)
  *  -  poly_degree
     -  -1
     -  [-1, 0, 1]
     -  ['int']
     -  -1 means no global polynomial, 0 means constant, 1 means linear trend
  *  -  data_dir
     -  None
     -  None
     -  ['str']
     -  Directory for loading / saving cached data; None means do not save or load
  *  -  reg
     -  1e-10
     -  None
     -  ['int', 'float']
     -  Regularization coeff.
  *  -  max_print_depth
     -  5
     -  None
     -  ['int']
     -  Maximum depth (level of nesting) to print operation descriptions and times
