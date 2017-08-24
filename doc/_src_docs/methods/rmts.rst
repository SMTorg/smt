Regularized minimal-energy tensor-product splines
=================================================

Regularized minimal-energy tensor-product splines (RMTS) is a type of surrogate model for
low-dimensional problems with large datasets and where fast prediction is desired.
The underlying mathematical functions are tensor-product splines,
which limits RMTS to up to 4-D problems, or 5-D problems in certain cases.
On the other hand, tensor-product splines enable a very fast prediction time
that does not increase with the number of training points.
Unlike other methods like Kriging and radial basis functions,
RMTS is not susceptible to numerical issues when there is a large number of training points
or when there are points that are too close together.

The prediction equation for RMTS is

.. math ::

  \newcommand\RR{\mathbb{R}}
  \newcommand\w{\mathbf{w}}
  \newcommand\H{\mathbf{H}}
  \newcommand\F{\mathbf{F}}
  \newcommand\y{\mathbf{y}}
  \newcommand\x{\mathbf{x}}
  \newcommand\yt{\mathbf{yt}}
  \newcommand\xt{\mathbf{xt}}
  \newcommand\sumt{\sum_i^{nt}}
  y = \F(\x) \w ,

where
:math:`\x \in \RR^{nx}` is the prediction input vector,
:math:`y \in \RR` is the prediction output,
:math:`\w \in \RR^{nw}` is the vector of spline coefficients,
and
:math:`\F(\x) \in \RR^{nw}` is the vector mapping the spline coefficients to the prediction output.

RMTS computes the coefficients of the splines, :math:`\w`, by solving an energy minimization problem
subject to the conditions that the splines pass through the training points.
This is formulated as an unconstrained optimization problem
where the objective function consists of a term containing the second derivatives of the splines,
another term representing the approximation error for the training points,
and another term for regularization:

.. math ::

  \begin{array}{r l}
    \underset{\w}{\min} & \frac{1}{2} \w^T \H \w
    + \frac{1}{2} \beta \w^T \w
    \\
    &
    + \frac{1}{2} \frac{1}{\alpha}
    \sumt \left[ \F(\xt_i) \w - \yt_i \right] ^ 2
  \end{array} ,

where
:math:`\xt_i \in \RR^{nx}` is the input vector for the :math:`i` th training point,
:math:`yt_i \in \RR` is the output value for the :math:`i` th training point,
:math:`\H \in \RR^{nw \times nw}` is the matrix containing the second derivatives,
:math:`\F(\xt_i) \in \RR^{nw}` is the vector mapping the spline coefficients to the :math:`i` th training output.
and :math:`\alpha` and :math:`\beta` are regularization coefficients.

In problems with a large number of training points relative to the number of spline coefficients,
the energy minimization term is not necessary;
this term can be zero-ed by setting the reg_cons option to zero.
In problems with a small dataset, the energy minimization is necessary.
When the true function has high curvature, the energy minimization can be counterproductive
in the regions of high curvature.
This can be addressed by increasing the quadratic approximation term to one of higher order,
and using Newton's method to solve the nonlinear system that results.
The nonlinear formulation is given by

.. math::

  \begin{array}{r l}
    \underset{\w}{\min} & \frac{1}{2} \w^T \H \w
    + \frac{1}{2} \beta \w^T \w
    \\
    &
    + \frac{1}{2} \frac{1}{\alpha}
    \sumt \left[ \F(\xt_i) \w - \yt_i \right] ^ p
  \end{array}
  ,

where :math:`p` is the order given by the approx_order option.
The number of Newton iterations can be specified via the :code:`nln_max_iter` option.

RMTS is implemented in SMT with two choices of splines:

1. B-splines (RMTB): RMTB uses B-splines with a uniform knot vector in each dimension.
The number of B-spline control points and the B-spline order in each dimension are options
that trade off efficiency and precision of the interpolant.

2. Cubic Hermite splines (RMTC): RMTC divides the domain into tensor-product cubic elements.
For adjacent elements, the values and derivatives are continuous.
The number of elements in each dimension is an option that trades off efficiency and precision.

In general, RMTB is the better choice when training time is the most important,
while RMTC is the better choice when accuracy of the interpolant is the most important.

Usage (RMTB)
------------

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
           Computing dof2coeff - done. Time (sec):  0.0000041
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0005860
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0019469
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0007401
        Pre-computing matrices - done. Time (sec):  0.0034020
        Solving for degrees of freedom ...
           Solving initial linear problem (n=20) ...
              Assembling linear system ...
              Assembling linear system - done. Time (sec):  0.0008399
              Initializing linear solver ...
              Initializing linear solver - done. Time (sec):  0.0000420
              Solving linear system (col. 0) ...
                 Running cg Krylov solver (20 x 20 mtx) ...
                 Running cg Krylov solver (20 x 20 mtx) - done. Time (sec):  0.0085273
              Solving linear system (col. 0) - done. Time (sec):  0.0085893
           Solving initial linear problem (n=20) - done. Time (sec):  0.0095351
           Solving nonlinear problem (col. 0) ...
              Nonlinear (itn, iy, grad. norm, func.) :   0   0 2.646068513e-15 1.135672323e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0020061
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000389
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0054872
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0007889
              Nonlinear (itn, iy, grad. norm, func.) :   1   0 2.646086053e-15 1.135672321e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0008359
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000310
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0021870
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0002282
              Nonlinear (itn, iy, grad. norm, func.) :   2   0 2.945795907e-14 1.135663464e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0008030
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000291
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0020173
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000432
              Nonlinear (itn, iy, grad. norm, func.) :   3   0 9.225368607e-14 1.121276602e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0014238
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000558
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0029612
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000451
              Nonlinear (itn, iy, grad. norm, func.) :   4   0 2.711738359e-14 1.119908145e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0008550
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000298
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0023921
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000463
              Nonlinear (itn, iy, grad. norm, func.) :   5   0 7.833904678e-15 1.119647315e-15
           Solving nonlinear problem (col. 0) - done. Time (sec):  0.0256481
        Solving for degrees of freedom - done. Time (sec):  0.0353043
     Training - done. Time (sec):  0.0393147
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003970
     
     Prediction time/pt. (sec) :  0.0000040
     
  
.. figure:: rmts.png
  :scale: 80 %
  :align: center

Usage (RMTC)
------------

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
           Computing dof2coeff - done. Time (sec):  0.0015290
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0005159
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0022368
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0010161
        Pre-computing matrices - done. Time (sec):  0.0054610
        Solving for degrees of freedom ...
           Solving initial linear problem (n=42) ...
              Assembling linear system ...
              Assembling linear system - done. Time (sec):  0.0008271
              Initializing linear solver ...
              Initializing linear solver - done. Time (sec):  0.0000429
              Solving linear system (col. 0) ...
                 Running cg Krylov solver (42 x 42 mtx) ...
                 Running cg Krylov solver (42 x 42 mtx) - done. Time (sec):  0.0047691
              Solving linear system (col. 0) - done. Time (sec):  0.0048261
           Solving initial linear problem (n=42) - done. Time (sec):  0.0057619
           Solving nonlinear problem (col. 0) ...
              Nonlinear (itn, iy, grad. norm, func.) :   0   0 3.799115482e-15 1.133573309e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0040293
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000603
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0246327
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0005522
              Nonlinear (itn, iy, grad. norm, func.) :   1   0 3.630563558e-15 1.133570797e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0008132
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000269
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0085530
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000448
              Nonlinear (itn, iy, grad. norm, func.) :   2   0 1.695886087e-14 1.117611568e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0008180
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000279
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0119858
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000751
              Nonlinear (itn, iy, grad. norm, func.) :   3   0 4.514073631e-15 1.117528217e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0014341
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000482
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0167441
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000830
              Nonlinear (itn, iy, grad. norm, func.) :   4   0 1.009913860e-15 1.117516752e-15
                 Assembling linear system ...
                 Assembling linear system - done. Time (sec):  0.0014002
                 Initializing linear solver ...
                 Initializing linear solver - done. Time (sec):  0.0000520
                 Solving linear system ...
                 Solving linear system - done. Time (sec):  0.0079188
                 Performing line search ...
                 Performing line search - done. Time (sec):  0.0000691
              Nonlinear (itn, iy, grad. norm, func.) :   5   0 1.587034176e-16 1.117515739e-15
           Solving nonlinear problem (col. 0) - done. Time (sec):  0.0835660
        Solving for degrees of freedom - done. Time (sec):  0.0894604
     Training - done. Time (sec):  0.0956481
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005507
     
     Prediction time/pt. (sec) :  0.0000055
     
  
.. figure:: rmts.png
  :scale: 80 %
  :align: center

Options (RMTB)
--------------

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

Options (RMTC)
--------------

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
