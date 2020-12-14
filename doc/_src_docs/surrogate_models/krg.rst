Kriging
=======

Kriging is an interpolating model that is a linear combination of a known function :math:`f_i({\bf x})` which is added to a realization of a stochastic process :math:`Z({\bf x})`

.. math ::
  \hat{y} = \sum\limits_{i=1}^k\beta_if_i({\bf x})+Z({\bf x}).

:math:`Z({\bf x})` is a realization of a stochastique process with mean zero and spatial covariance function given by

.. math ::
  cov\left[Z\left({\bf x}^{(i)}\right),Z\left({\bf x}^{(j)}\right)\right] =\sigma^2R\left({\bf x}^{(i)},{\bf x}^{(j)}\right)

where :math:`\sigma^2` is the process variance, and :math:`R` is the correlation.
Two types of correlation functions are available in SMT: the exponential (Ornstein-Uhlenbeck process) and Gaussian correlation functions

.. math ::
  \prod\limits_{l=1}^{nx}\exp\left(-\theta_l\left|x_l^{(i)}-x_l^{(j)}\right|\right),\qquad \qquad \qquad\prod\limits_{l=1}^{nx}\exp\left(-\theta_l\left(x_l^{(i)}-x_l^{(j)}\right)^{2}\right) \quad \forall\ \theta_l\in\mathbb{R}^+\\
  \text{Exponential correlation function} \quad \qquad\text{Gaussian correlation function}\qquad \qquad

These two correlation functions are called by 'abs_exp' (exponential) and 'squar_exp' (Gaussian) in SMT.

The deterministic term :math:`\sum\limits_{i=1}^k\beta_i f_i({\bf x})` can be replaced by a constant, a linear model, or a quadratic model.
These three types are available in SMT.

More details about the kriging approach could be found in [1]_.

Kriging with categorical or integer variables 
---------------------------------------------

The goal is to be able to build a model for mixed typed variables. 
This algorithm has been presented by  Garrido-Merchán and Hernández-Lobato in 2020 [2]_.

To incorporate integer (with order relation) and categorical variables (with no order), we used continuous relaxation.
For integer, we add a continuous dimension with the same bounds and then we round in the prediction to the closer integer.
For categorical, we add as many continuous dimensions with bounds [0,1] as possible output values for the variable and 
then we round in the prediction to the output dimension giving the greatest continuous prediction.

More details available in [2]_. See also :ref:`Mixed-Integer Sampling and Surrogate`.

Implementation Note: Mixed variables handling is available for all kriging models (KRG, KPLS or KPLSK) but cannot be used with derivatives computation.

.. [1] Sacks, J. and Schiller, S. B. and Welch, W. J., Designs for computer experiments, Technometrics 31 (1) (1989) 41--47.

.. [2] E. C. Garrido-Merchan and D. Hernandez-Lobato, Dealing with categorical and integer-valued variables in Bayesian Optimization with Gaussian processes, Neurocomputing 380 (2020) 20-–35.

Usage
-----

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.surrogate_models import KRG
  
  xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
  yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])
  
  sm = KRG(theta0=[1e-2])
  sm.set_training_values(xt, yt)
  sm.train()
  
  num = 100
  x = np.linspace(0.0, 4.0, num)
  y = sm.predict_values(x)
  
  plt.plot(xt, yt, "o")
  plt.plot(x, y)
  plt.xlabel("x")
  plt.ylabel("y")
  plt.legend(["Training data", "Prediction"])
  plt.show()
  
::

  ___________________________________________________________________________
     
                                    Kriging
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 5
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
     Training - done. Time (sec):  0.0000000
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: krg_Test_test_krg.png
  :scale: 80 %
  :align: center

Usage with mixed variables
--------------------------

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.surrogate_models import KRG
  from smt.applications.mixed_integer import MixedIntegerSurrogateModel, INT
  
  xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
  yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])
  
  # xtypes = [FLOAT, INT, (ENUM, 3), (ENUM, 2)]
  # FLOAT means x1 continuous
  # INT means x2 integer
  # (ENUM, 3) means x3, x4 & x5 are 3 levels of the same categorical variable
  # (ENUM, 2) means x6 & x7 are 2 levels of the same categorical variable
  
  sm = MixedIntegerSurrogateModel(
      xtypes=[INT], xlimits=[[0, 4]], surrogate=KRG(theta0=[1e-2])
  )
  sm.set_training_values(xt, yt)
  sm.train()
  
  num = 100
  x = np.linspace(0.0, 4.0, num)
  y = sm.predict_values(x)
  
  plt.plot(xt, yt, "o")
  plt.plot(x, y)
  plt.xlabel("x")
  plt.ylabel("y")
  plt.legend(["Training data", "Prediction"])
  plt.show()
  
::

  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: krg_Test_test_mixed_int_krg.png
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
  *  -  poly
     -  constant
     -  ['constant', 'linear', 'quadratic']
     -  ['str']
     -  Regression function type
  *  -  corr
     -  squar_exp
     -  ['abs_exp', 'squar_exp', 'act_exp', 'matern52', 'matern32']
     -  ['str']
     -  Correlation function type
  *  -  theta0
     -  [0.01]
     -  None
     -  ['list', 'ndarray']
     -  Initial hyperparameters
  *  -  hyper_opt
     -  Cobyla
     -  ['Cobyla', 'TNC']
     -  ['str']
     -  Optimiser for hyperparameters optimisation
  *  -  eval_noise
     -  False
     -  [True, False]
     -  ['bool']
     -  noise evaluation flag
  *  -  noise0
     -  1e-06
     -  None
     -  ['float', 'list']
     -  Initial noise hyperparameter
