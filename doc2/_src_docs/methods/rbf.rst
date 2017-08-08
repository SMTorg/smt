Radial basis functions
======================

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
           Performing LU fact. (5 x 5 mtx) - done. Time (sec):  0.0000870
        Initializing linear solver - done. Time (sec):  0.0001230
        Solving linear system (col. 0) ...
           Back solving (5 x 5 mtx) ...
           Back solving (5 x 5 mtx) - done. Time (sec):  0.0000744
        Solving linear system (col. 0) - done. Time (sec):  0.0001032
     Training - done. Time (sec):  0.0005929
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000367
     
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
     -  Acceptable values
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
