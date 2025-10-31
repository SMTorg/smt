.. _cckrg-ref-label:

Cooperative Components Kriging (CoopCompKRG)
============================================

Cooperative Components Kriging is a way of fitting a high-dimensional ordinary Kriging model by sequential lower-dimensional component model fits. For each component, only the associated hyperparameters are optimized. All other hyperparameters are set to a so-called cooperative context vector, which contains the current best hyperparameter values.

This application contains the single component model fits. The loop over the components has to be implemented individually, as shown in the usage example below.

The cooperative components model fit was developed as part of a high-dimensional surrogate-based optimization process. It is inspired by distributed multi-disciplinary design optimization (MDO) approaches and the cooperative EGO by Zhan et al. [1]_.


References
----------
.. [1] Zhan, D., Wu, J., Xing, H. et al., A cooperative approach to efficient global optimization. J Glob Optim 88, 327–357 (2024).

Usage
-----

.. code-block:: python

  import random
  
  import numpy as np
  
  from smt.applications import CoopCompKRG
  from smt.problems import TensorProduct
  from smt.sampling_methods import LHS
  
  # The problem is the exponential problem with dimension 10
  ndim = 10
  prob = TensorProduct(ndim=ndim, func="exp")
  
  # Example with three random components
  # (use physical components if available)
  ncomp = 3
  
  # Initial sampling
  samp = LHS(xlimits=prob.xlimits, seed=42)
  xt = samp(50)
  yt = prob(xt)
  
  # Random design variable to component allocation
  comps = [*range(ncomp)]
  vars = [*range(ndim)]
  random.shuffle(vars)
  comp_var = np.full((ndim, ncomp), False)
  for c in comps:
      comp_size = int(ndim / ncomp)
      start = c * comp_size
      end = (c + 1) * comp_size
      if c + 1 == ncomp:
          end = max((c + 1) * comp_size, ndim)
      comp_var[vars[start:end], c] = True
  
  # Cooperative components Kriging model fit
  sm = CoopCompKRG()
  for active_coop_comp in comps:
      sm.set_training_values(xt, yt)
      sm.train(active_coop_comp, comp_var)
  
  # Prediction as for ordinary Kriging
  xpoint = (-5 + np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])) / 10.0
  print(sm.predict_values(xpoint))
  print(sm.predict_variances(xpoint))
  
::

  ___________________________________________________________________________
     
                        Cooperative Components Kriging
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 50
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
     Training - done. Time (sec):  0.1202128
  ___________________________________________________________________________
     
                        Cooperative Components Kriging
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 50
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
     Training - done. Time (sec):  0.1307766
  ___________________________________________________________________________
     
                        Cooperative Components Kriging
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 50
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
     Training - done. Time (sec):  0.1591473
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 1
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  [[-2.29343475]]
  [[32.46448918]]
  

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
     -  ['str']
     -  Regression function type
  *  -  corr
     -  squar_exp
     -  ['pow_exp', 'abs_exp', 'squar_exp', 'squar_sin_exp', 'matern52', 'matern32']
     -  ['str']
     -  Correlation function type
  *  -  pow_exp_power
     -  1.9
     -  None
     -  ['float']
     -  Power for the pow_exp kernel function (valid values in (0.0, 2.0]).                 This option is set automatically when corr option is squar, abs, or matern.
  *  -  categorical_kernel
     -  MixIntKernelType.CONT_RELAX
     -  [<MixIntKernelType.CONT_RELAX: 'CONT_RELAX'>, <MixIntKernelType.GOWER: 'GOWER'>, <MixIntKernelType.EXP_HOMO_HSPHERE: 'EXP_HOMO_HSPHERE'>, <MixIntKernelType.HOMO_HSPHERE: 'HOMO_HSPHERE'>, <MixIntKernelType.COMPOUND_SYMMETRY: 'COMPOUND_SYMMETRY'>]
     -  None
     -  The kernel to use for categorical inputs. Only for non continuous Kriging
  *  -  hierarchical_kernel
     -  MixHrcKernelType.ALG_KERNEL
     -  [<MixHrcKernelType.ALG_KERNEL: 'ALG_KERNEL'>, <MixHrcKernelType.ARC_KERNEL: 'ARC_KERNEL'>]
     -  None
     -  The kernel to use for mixed hierarchical inputs. Only for non continuous Kriging
  *  -  nugget
     -  2.220446049250313e-14
     -  None
     -  ['float']
     -  a jitter for numerical stability
  *  -  theta0
     -  [0.01]
     -  None
     -  ['list', 'ndarray']
     -  Initial hyperparameters
  *  -  theta_bounds
     -  [1e-06, 20.0]
     -  None
     -  ['list', 'ndarray']
     -  bounds for hyperparameters
  *  -  hyper_opt
     -  Cobyla
     -  ['Cobyla']
     -  ['str']
     -  Correlation function type
  *  -  eval_noise
     -  False
     -  [True, False]
     -  ['bool']
     -  noise evaluation flag
  *  -  noise0
     -  [0.0]
     -  None
     -  ['list', 'ndarray']
     -  Initial noise hyperparameters
  *  -  noise_bounds
     -  [2.220446049250313e-14, 10000000000.0]
     -  None
     -  ['list', 'ndarray']
     -  bounds for noise hyperparameters
  *  -  use_het_noise
     -  False
     -  [True, False]
     -  ['bool']
     -  heteroscedastic noise evaluation flag
  *  -  n_start
     -  10
     -  None
     -  ['int']
     -  number of optimizer runs (multistart method)
  *  -  xlimits
     -  None
     -  None
     -  ['list', 'ndarray']
     -  definition of a design space of float (continuous) variables: array-like of size nx x 2 (lower, upper bounds)
  *  -  design_space
     -  None
     -  None
     -  ['BaseDesignSpace', 'list', 'ndarray']
     -  definition of the (hierarchical) design space: use `smt.design_space.DesignSpace` as the main API. Also accepts list of float variable bounds
  *  -  is_ri
     -  False
     -  None
     -  ['bool']
     -  activate reinterpolation for noisy cases
  *  -  seed
     -  41
     -  None
     -  ['NoneType', 'int', 'Generator']
     -  Numpy Generator object or seed number which controls random draws                 for internal optim (set by default to get reproductibility)
  *  -  random_state
     -  None
     -  None
     -  ['NoneType', 'int', 'RandomState']
     -  DEPRECATED (use seed instead): Numpy RandomState object or seed number which controls random draws                 for internal optim (set by default to get reproductibility)
