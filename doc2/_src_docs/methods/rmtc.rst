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
           Computing dof2coeff - done. Time (sec):  0.0008330
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0004590
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0011089
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0005059
        Pre-computing matrices - done. Time (sec):  0.0029759
        Solving for degrees of freedom ...
           Solving initial linear problem (n=42) ...
              Assembling linear system ...
              Assembling linear system - done. Time (sec):  0.0004511
              Initializing linear solver ...
              Initializing linear solver - done. Time (sec):  0.0000200
              Solving linear system (col. 0) ...
                 Running cg Krylov solver (42 x 42 mtx) ...
                 Running cg Krylov solver (42 x 42 mtx) - done. Time (sec):  0.0023010
              Solving linear system (col. 0) - done. Time (sec):  0.0023239
           Solving initial linear problem (n=42) - done. Time (sec):  0.0028248
           Solving nonlinear problem (col. 0) ...
              Nonlinear (itn, iy, grad. norm, func.) :   0   0 3.799115482e-15 1.133573309e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0008628
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000172
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0061972
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0005260
              Nonlinear (itn, iy, grad. norm, func.) :   1   0 3.630563558e-15 1.133570797e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0007820
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000219
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0055821
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000420
              Nonlinear (itn, iy, grad. norm, func.) :   2   0 1.695886087e-14 1.117611568e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0007811
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000210
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0041389
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000410
              Nonlinear (itn, iy, grad. norm, func.) :   3   0 4.514073631e-15 1.117528217e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0007839
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000191
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0065899
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000730
              Nonlinear (itn, iy, grad. norm, func.) :   4   0 1.009913860e-15 1.117516752e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0015681
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000367
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0067861
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0001340
              Nonlinear (itn, iy, grad. norm, func.) :   5   0 1.587034176e-16 1.117515739e-15
           Solving nonlinear problem (col. 0) - done. Time (sec):  0.0382249
        Solving for degrees of freedom - done. Time (sec):  0.0411179
     Training - done. Time (sec):  0.0444541
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007250
     
     Prediction time/pt. (sec) :  0.0000073
     
  
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
  *  -  num_elements
     -  4
     -  None
     -  ['Integral', 'list', 'ndarray']
     -  # elements in each dimension - ndarray [nx]
