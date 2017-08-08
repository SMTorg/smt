Second-order polynomial approximation
=====================================

The square polynomial model can be expressed by

.. math ::
  {\bf y} = {\bf X\beta} + {\bf \epsilon},

where :math:`{\bf \epsilon}` is a vector of random errors and

.. math ::
  {\bf X} =
  \begin{bmatrix}
      1&x_{1}^{(1)} & \dots&x_{d}^{(1)} & x_{1}^{(1)}x_{2}^{(1)} & \dots  & x_{d-1}^{(1)}x_{d}^{(1)}&{x_{1}^{(1)}}^2 & \dots&{x_{
      d}^{(1)}}^2 \\
      \vdots&\vdots & \dots&\vdots & \vdots & \dots  & \vdots&\vdots & \vdots\\
      1&x_{1}^{(n)} & \dots&x_{d}^{(n)} & x_{1}^{(n)}x_{2}^{(n)} & \dots  & x_{d-1}^{(n)}x_{d}^{(n)}&{x_{1}^{(n)}}^2 & \dots&{x_{
      d}^{(n)}}^2 \\
  \end{bmatrix}.

The vector of estimated polynomial regression coefficients using ordinary least square estimation is

.. math ::
  {\bf \beta} = {\bf X^TX}^{-1} {\bf X^Ty}.

Usage
-----

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.methods import PA2
  
  xt = np.array([0., 1., 2., 3., 4.])
  yt = np.array([0., 1., 1.5, 0.5, 1.0])
  
  sm = PA2()
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
     
                                      PA2
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 5
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
     Training - done. Time (sec):  0.0002239
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000288
     
     Prediction time/pt. (sec) :  0.0000003
     
  
.. figure:: pa2.png
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
  *  -  data_dir
     -  None
     -  None
     -  ['str']
     -  Directory for loading / saving cached data; None means do not save or load
