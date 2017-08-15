Regularized minimal-energy tensor-product B-splines
===================================================

Usage
-----

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
           Computing dof2coeff - done. Time (sec):  0.0000029
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0005379
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0018520
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0006721
        Pre-computing matrices - done. Time (sec):  0.0031900
        Solving for degrees of freedom ...
           Solving initial linear problem (n=20) ...
              Assembling linear system ...
              Assembling linear system - done. Time (sec):  0.0007939
              Initializing linear solver ...
              Initializing linear solver - done. Time (sec):  0.0000291
              Solving linear system (col. 0) ...
                 Running cg Krylov solver (20 x 20 mtx) ...
                 Running cg Krylov solver (20 x 20 mtx) - done. Time (sec):  0.0035279
              Solving linear system (col. 0) - done. Time (sec):  0.0035758
           Solving initial linear problem (n=20) - done. Time (sec):  0.0044692
           Solving nonlinear problem (col. 0) ...
              Nonlinear (itn, iy, grad. norm, func.) :   0   0 2.645720108e-15 1.135812944e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0013261
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000238
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0089669
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0010738
              Nonlinear (itn, iy, grad. norm, func.) :   1   0 2.715444789e-15 1.135810436e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0012710
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000260
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0042229
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0001321
              Nonlinear (itn, iy, grad. norm, func.) :   2   0 9.147277374e-14 1.125019496e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0012600
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000250
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0043261
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0001409
              Nonlinear (itn, iy, grad. norm, func.) :   3   0 7.664218198e-14 1.121922545e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0012639
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000260
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0041311
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000770
              Nonlinear (itn, iy, grad. norm, func.) :   4   0 2.239826012e-14 1.119834510e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0012619
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000260
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0040290
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000749
              Nonlinear (itn, iy, grad. norm, func.) :   5   0 6.282080603e-15 1.119630128e-15
           Solving nonlinear problem (col. 0) - done. Time (sec):  0.0377779
        Solving for degrees of freedom - done. Time (sec):  0.0423470
     Training - done. Time (sec):  0.0462751
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005291
     
     Prediction time/pt. (sec) :  0.0000053
     
  
.. figure:: rmtb.png
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
  *  -  min_energy
     -  True
     -  None
     -  ['bool']
     -  Whether to perform energy minimization
  *  -  reg_dv
     -  1e-10
     -  None
     -  ['Integral', 'float']
     -  Regularization coeff. for system degrees of freedom. This ensures there is always a unique solution
  *  -  mtx_free
     -  False
     -  None
     -  ['bool']
     -  Whether to solve the linear system in a matrix-free way
  *  -  print_prediction
     -  True
     -  None
     -  ['bool']
     -  Whether to print prediction information
  *  -  num_ctrl_pts
     -  15
     -  None
     -  ['Integral', 'tuple', 'list', 'ndarray']
     -  # B-spline control points in each dimension - length [nx]
  *  -  data_dir
     -  None
     -  [None]
     -  ['str']
     -  Directory for loading / saving cached data; None means do not save or load
  *  -  print_solver
     -  True
     -  None
     -  ['bool']
     -  Whether to print solver information
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
  *  -  extrapolate
     -  False
     -  None
     -  ['bool']
     -  Whether to perform linear extrapolation for external evaluation points
  *  -  save_energy_terms
     -  False
     -  None
     -  ['bool']
     -  Whether to cache energy terms in the data_dir directory
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
  *  -  approx_order
     -  4
     -  None
     -  ['Integral']
     -  Exponent in the approximation term
  *  -  reg_cons
     -  0.0001
     -  None
     -  ['Integral', 'float']
     -  Negative of the regularization coeff. of the Lagrange mult. block The weight of the energy terms (and reg_dv) relative to the approx terms
  *  -  smoothness
     -  1.0
     -  None
     -  ['Integral', 'float', 'tuple', 'list', 'ndarray']
     -  Smoothness parameter in each dimension - length nx. None implies uniform
  *  -  xlimits
     -  None
     -  None
     -  ['ndarray']
     -  Lower/upper bounds in each dimension - ndarray [nx, 2]
  *  -  solver
     -  krylov
     -  ['krylov-dense', 'dense-lu', 'dense-chol', 'lu', 'ilu', 'krylov', 'krylov-lu', 'krylov-mg', 'gs', 'jacobi', 'mg', 'null']
     -  ['LinearSolver']
     -  Linear solver
  *  -  max_print_depth
     -  5
     -  None
     -  ['Integral']
     -  Maximum depth (level of nesting) to print operation descriptions and times
  *  -  grad_weight
     -  0.5
     -  None
     -  ['Integral', 'float']
     -  Weight on gradient training data
  *  -  print_training
     -  True
     -  None
     -  ['bool']
     -  Whether to print training information
  *  -  order
     -  3
     -  None
     -  ['Integral', 'tuple', 'list', 'ndarray']
     -  B-spline order in each dimension - length [nx]
