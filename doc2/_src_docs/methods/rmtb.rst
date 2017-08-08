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
  plt.ylabel('y')
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
           Computing dof2coeff - done. Time (sec):  0.0000031
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0003281
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0010190
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0003986
        Pre-computing matrices - done. Time (sec):  0.0018079
        Solving for degrees of freedom ...
           Solving initial linear problem (n=20) ...
              Assembling linear system ...
              Assembling linear system - done. Time (sec):  0.0004401
              Initializing linear solver ...
              Initializing linear solver - done. Time (sec):  0.0000188
              Solving linear system (col. 0) ...
                 Running cg Krylov solver (20 x 20 mtx) ...
                 Running cg Krylov solver (20 x 20 mtx) - done. Time (sec):  0.0019829
              Solving linear system (col. 0) - done. Time (sec):  0.0020041
           Solving initial linear problem (n=20) - done. Time (sec):  0.0024929
           Solving nonlinear problem (col. 0) ...
              Nonlinear (itn, iy, grad. norm, func.) :   0   0 2.646068513e-15 1.135672323e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0007591
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000150
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0057459
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0009580
              Nonlinear (itn, iy, grad. norm, func.) :   1   0 2.958916346e-15 1.135667790e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0014989
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000510
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0051060
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0001938
              Nonlinear (itn, iy, grad. norm, func.) :   2   0 1.124756199e-13 1.130347200e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0008492
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000272
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0022070
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000410
              Nonlinear (itn, iy, grad. norm, func.) :   3   0 3.301153444e-14 1.119917820e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0007770
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000269
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0021367
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000670
              Nonlinear (itn, iy, grad. norm, func.) :   4   0 9.406540080e-15 1.119646967e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0008388
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000319
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0032670
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000689
              Nonlinear (itn, iy, grad. norm, func.) :   5   0 7.257840804e-15 1.119637838e-15
           Solving nonlinear problem (col. 0) - done. Time (sec):  0.0280340
        Solving for degrees of freedom - done. Time (sec):  0.0305870
     Training - done. Time (sec):  0.0327220
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003569
     
     Prediction time/pt. (sec) :  0.0000036
     
  
.. figure:: rmtb.png
  :scale: 80 %
  :align: center

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
  *  -  xlimits
     -  None
     -  None
     -  ['ndarray']
     -  Lower/upper bounds in each dimension - ndarray [nx, 2]
  *  -  smoothness
     -  1.0
     -  None
     -  ['Integral', 'float', 'tuple', 'list', 'ndarray']
     -  Smoothness parameter in each dimension - length nx. None implies uniform
  *  -  reg_dv
     -  1e-10
     -  None
     -  ['Integral', 'float']
     -  Regularization coeff. for system degrees of freedom. This ensures there is always a unique solution
  *  -  reg_cons
     -  0.0001
     -  None
     -  ['Integral', 'float']
     -  Negative of the regularization coeff. of the Lagrange mult. block The weight of the energy terms (and reg_dv) relative to the approx terms
  *  -  extrapolate
     -  False
     -  None
     -  ['bool']
     -  Whether to perform linear extrapolation for external evaluation points
  *  -  min_energy
     -  True
     -  None
     -  ['bool']
     -  Whether to perform energy minimization
  *  -  approx_order
     -  4
     -  None
     -  ['Integral']
     -  Exponent in the approximation term
  *  -  mtx_free
     -  False
     -  None
     -  ['bool']
     -  Whether to solve the linear system in a matrix-free way
  *  -  solver
     -  krylov
     -  ['krylov-dense', 'dense-lu', 'dense-chol', 'lu', 'ilu', 'krylov', 'krylov-lu', 'krylov-mg', 'gs', 'jacobi', 'mg', 'null']
     -  ['LinearSolver']
     -  Linear solver
  *  -  grad_weight
     -  0.5
     -  None
     -  ['Integral', 'float']
     -  Weight on gradient training data
  *  -  nln_max_iter
     -  5
     -  None
     -  ['Integral']
     -  maximum number of nonlinear iterations
  *  -  line_search
     -  backtracking
     -  ['backtracking', 'bracketed', 'quadratic', 'cubic', 'null']
     -  ['LineSearch']
     -  Line search algorithm
  *  -  save_energy_terms
     -  False
     -  None
     -  ['bool']
     -  Whether to cache energy terms in the data_dir directory
  *  -  data_dir
     -  None
     -  [None]
     -  ['str']
     -  Directory for loading / saving cached data; None means do not save or load
  *  -  max_print_depth
     -  5
     -  None
     -  ['Integral']
     -  Maximum depth (level of nesting) to print operation descriptions and times
  *  -  order
     -  3
     -  None
     -  ['Integral', 'tuple', 'list', 'ndarray']
     -  B-spline order in each dimension - length [nx]
  *  -  num_ctrl_pts
     -  15
     -  None
     -  ['Integral', 'tuple', 'list', 'ndarray']
     -  # B-spline control points in each dimension - length [nx]
