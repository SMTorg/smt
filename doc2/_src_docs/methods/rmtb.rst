Regularized minimal-energy tensor-product B-splines
===================================================

.. code-block:: python

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
  plt.ylabel('x')
  plt.legend(['Training data', 'Prediction'])
  plt.show()
  
.. plot::

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
  plt.ylabel('x')
  plt.legend(['Training data', 'Prediction'])
  plt.show()
  
::

  ___________________________________________________________________________
     
                                     RMTB
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 5
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Pre-computing matrices ...
           Computing dof2coeff ...
           Computing dof2coeff - done. Time (sec):  0.0000050
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0007010
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0021167
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0008032
        Pre-computing matrices - done. Time (sec):  0.0037611
        Solving for degrees of freedom ...
           Solving initial linear problem (n=20) ...
              Assembling linear system ...
              Assembling linear system - done. Time (sec):  0.0009050
              Initializing linear solver ...
              Initializing linear solver - done. Time (sec):  0.0000410
              Solving linear system (col. 0) ...
                 Running cg Krylov solver (20 x 20 mtx) ...
                 Running cg Krylov solver (20 x 20 mtx) - done. Time (sec):  0.0062368
              Solving linear system (col. 0) - done. Time (sec):  0.0062861
           Solving initial linear problem (n=20) - done. Time (sec):  0.0072930
           Solving nonlinear problem (col. 0) ...
              Nonlinear (itn, iy, grad. norm, func.) :   0   0 2.646675829e-15 1.135884197e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0014331
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000391
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0053792
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0005322
              Nonlinear (itn, iy, grad. norm, func.) :   1   0 2.645350434e-15 1.135864523e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0015388
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000441
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0084422
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0002930
              Nonlinear (itn, iy, grad. norm, func.) :   2   0 1.031646202e-13 1.121194897e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0029960
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0001032
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0099111
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0002770
              Nonlinear (itn, iy, grad. norm, func.) :   3   0 6.290036340e-14 1.120519659e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0013249
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000651
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0028071
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000460
              Nonlinear (itn, iy, grad. norm, func.) :   4   0 1.842014446e-14 1.119761776e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0007989
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000248
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0023050
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000420
              Nonlinear (itn, iy, grad. norm, func.) :   5   0 5.266062400e-15 1.119620437e-15
           Solving nonlinear problem (col. 0) - done. Time (sec):  0.0444870
        Solving for degrees of freedom - done. Time (sec):  0.0518830
     Training - done. Time (sec):  0.0562963
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003731
     
     Prediction time/pt. (sec) :  0.0000037
     
  

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
     -  [None]
     -  ['bool']
     -  Global print toggle. If False, all printing is suppressed
  *  -  print_training
     -  True
     -  [None]
     -  ['bool']
     -  Whether to print training information
  *  -  print_prediction
     -  True
     -  [None]
     -  ['bool']
     -  Whether to print prediction information
  *  -  print_problem
     -  True
     -  [None]
     -  ['bool']
     -  Whether to print problem information
  *  -  print_solver
     -  True
     -  [None]
     -  ['bool']
     -  Whether to print solver information
  *  -  xlimits
     -  None
     -  [None]
     -  ['ndarray']
     -  Lower/upper bounds in each dimension - ndarray [nx, 2]
  *  -  smoothness
     -  1.0
     -  [None]
     -  ['Integral', 'float', 'tuple', 'list', 'ndarray']
     -  Smoothness parameter in each dimension - length nx. None implies uniform
  *  -  reg_dv
     -  1e-10
     -  [None]
     -  ['Integral', 'float']
     -  Regularization coeff. for system degrees of freedom. This ensures there is always a unique solution
  *  -  reg_cons
     -  0.0001
     -  [None]
     -  ['Integral', 'float']
     -  Negative of the regularization coeff. of the Lagrange mult. block The weight of the energy terms (and reg_dv) relative to the approx terms
  *  -  extrapolate
     -  False
     -  [None]
     -  ['bool']
     -  Whether to perform linear extrapolation for external evaluation points
  *  -  min_energy
     -  True
     -  [None]
     -  ['bool']
     -  Whether to perform energy minimization
  *  -  approx_order
     -  4
     -  [None]
     -  ['Integral']
     -  Exponent in the approximation term
  *  -  mtx_free
     -  False
     -  [None]
     -  ['bool']
     -  Whether to solve the linear system in a matrix-free way
  *  -  solver
     -  krylov
     -  ['krylov-dense', 'dense-lu', 'dense-chol', 'lu', 'ilu', 'krylov', 'krylov-lu', 'krylov-mg', 'gs', 'jacobi', 'mg', 'null']
     -  ['LinearSolver']
     -  Linear solver
  *  -  grad_weight
     -  0.5
     -  [None]
     -  ['Integral', 'float']
     -  Weight on gradient training data
  *  -  nln_max_iter
     -  5
     -  [None]
     -  ['Integral']
     -  maximum number of nonlinear iterations
  *  -  line_search
     -  backtracking
     -  ['backtracking', 'bracketed', 'quadratic', 'cubic', 'null']
     -  ['LineSearch']
     -  Line search algorithm
  *  -  save_energy_terms
     -  False
     -  [None]
     -  ['bool']
     -  Whether to cache energy terms in the data_dir directory
  *  -  data_dir
     -  None
     -  [None]
     -  ['str']
     -  Directory for loading / saving cached data; None means do not save or load
  *  -  max_print_depth
     -  5
     -  [None]
     -  ['Integral']
     -  Maximum depth (level of nesting) to print operation descriptions and times
  *  -  order
     -  3
     -  [None]
     -  ['Integral', 'tuple', 'list', 'ndarray']
     -  B-spline order in each dimension - length [nx]
  *  -  num_ctrl_pts
     -  15
     -  [None]
     -  ['Integral', 'tuple', 'list', 'ndarray']
     -  # B-spline control points in each dimension - length [nx]
