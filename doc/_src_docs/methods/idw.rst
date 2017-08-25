Inverse-distance weighting
==========================

The inverse distance weighting [1]_ (IDW) model is an interpolating method
and the unknown points are calculated with a weighted average of the sampling points.

The prediction equation for IDW is

.. math ::

  \newcommand\RR{\mathbb{R}}
  \newcommand\y{\mathbf{y}}
  \newcommand\x{\mathbf{x}}
  \newcommand\yt{\mathbf{yt}}
  \newcommand\xt{\mathbf{xt}}
  \newcommand\sumt{\sum_i^{nt}}
  y =
  \left\{
  \begin{array}{ll}
    \frac{\sumt \beta(\x, \xt_i) \yt_i}{\sumt \beta(\x, \xt_i)},
    & \text{if} \quad \x \neq \xt_i \quad \forall i\\
    \yt_i
    & \text{if} \quad \x = \xt_i \quad \text{for some} \; i\\
  \end{array}
  \right. ,

where
:math:`\x \in \RR^{nx}` is the prediction input vector,
:math:`y \in \RR` is the prediction output,
:math:`\xt_i \in \RR^{nx}` is the input vector for the :math:`i` th training point,
and
:math:`yt_i \in \RR` is the output value for the :math:`i` th training point.
The weighting function :math:`\beta` is defined by

.. math ::

  \beta( \x_i , \x_j ) = || \x_i - \x_j ||_2 ^ {-p} ,

where :math:`p` a positive real number, called the power parameter.
This parameter must be strictly greater than 1 for the derivatives to be continuous.

.. [1] Shepard, D., A Two-dimensional Interpolation Function for Irregularly-spaced Data, Proceedings of the 1968 23rd ACM National Conference, 1968, pp. 517--524.

Usage
-----

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.methods import IDW
  
  xt = np.array([0., 1., 2., 3., 4.])
  yt = np.array([0., 1., 1.5, 0.5, 1.0])
  
  sm = IDW(p=2)
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
     
                                      IDW
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 5
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
     Training - done. Time (sec):  0.0002000
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000339
     
     Prediction time/pt. (sec) :  0.0000003
     
  
.. figure:: idw.png
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
  *  -  data_dir
     -  None
     -  None
     -  ['str']
     -  Directory for loading / saving cached data; None means do not save or load
  *  -  print_solver
     -  True
     -  None
     -  ['bool']
     -  Whether to print solver information
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
  *  -  p
     -  2.5
     -  None
     -  ['int', 'float']
     -  order of distance norm
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
