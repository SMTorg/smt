Regularized minimal-energy tensor-product cubic splines
=======================================================

Usage
-----

.. code-block:: python

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
  
::

  ___________________________________________________________________________
     
                                     RMTC
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 5
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Pre-computing matrices ...
           Computing dof2coeff ...
           Computing dof2coeff - done. Time (sec):  0.0018160
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0003531
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0015430
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0007081
        Pre-computing matrices - done. Time (sec):  0.0045378
        Solving for degrees of freedom ...
           Solving initial linear problem (n=42) ...
              Assembling linear system ...
              Assembling linear system - done. Time (sec):  0.0006311
              Initializing linear solver ...
              Initializing linear solver - done. Time (sec):  0.0000350
              Solving linear system (col. 0) ...
                 Running cg Krylov solver (42 x 42 mtx) ...
                 Running cg Krylov solver (42 x 42 mtx) - done. Time (sec):  0.0036631
              Solving linear system (col. 0) - done. Time (sec):  0.0037019
           Solving initial linear problem (n=42) - done. Time (sec):  0.0044279
           Solving nonlinear problem (col. 0) ...
              Nonlinear (itn, iy, grad. norm, func.) :   0   0 3.799115482e-15 1.133573309e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0011010
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000188
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0075970
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0008669
              Nonlinear (itn, iy, grad. norm, func.) :   1   0 3.443643778e-15 1.133567021e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0010550
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000200
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0070801
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000739
              Nonlinear (itn, iy, grad. norm, func.) :   2   0 1.760867582e-14 1.117593275e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0010412
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000200
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0060930
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000632
              Nonlinear (itn, iy, grad. norm, func.) :   3   0 4.675212285e-15 1.117525470e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0010419
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000210
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0059471
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000620
              Nonlinear (itn, iy, grad. norm, func.) :   4   0 9.728741606e-16 1.117516322e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0010481
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000198
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0057292
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000632
              Nonlinear (itn, iy, grad. norm, func.) :   5   0 9.505451733e-17 1.117515709e-15
           Solving nonlinear problem (col. 0) - done. Time (sec):  0.0423472
        Solving for degrees of freedom - done. Time (sec):  0.0468650
     Training - done. Time (sec):  0.0519900
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0004268
     
     Prediction time/pt. (sec) :  0.0000043
     
  
.. figure:: rmtc.png
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
  *  -  num_elements
     -  4
     -  None
     -  ['Integral', 'list', 'ndarray']
     -  # elements in each dimension - ndarray [nx]
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
  *  -  print_solver
     -  True
     -  None
     -  ['bool']
     -  Whether to print solver information
  *  -  solver
     -  krylov
     -  ['krylov-dense', 'dense-lu', 'dense-chol', 'lu', 'ilu', 'krylov', 'krylov-lu', 'krylov-mg', 'gs', 'jacobi', 'mg', 'null']
     -  ['LinearSolver']
     -  Linear solver
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
  *  -  max_print_depth
     -  5
     -  None
     -  ['Integral']
     -  Maximum depth (level of nesting) to print operation descriptions and times
  *  -  reg_dv
     -  1e-10
     -  None
     -  ['Integral', 'float']
     -  Regularization coeff. for system degrees of freedom. This ensures there is always a unique solution
  *  -  extrapolate
     -  False
     -  None
     -  ['bool']
     -  Whether to perform linear extrapolation for external evaluation points
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
  *  -  min_energy
     -  True
     -  None
     -  ['bool']
     -  Whether to perform energy minimization
  *  -  grad_weight
     -  0.5
     -  None
     -  ['Integral', 'float']
     -  Weight on gradient training data
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
  *  -  print_training
     -  True
     -  None
     -  ['bool']
     -  Whether to print training information
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
