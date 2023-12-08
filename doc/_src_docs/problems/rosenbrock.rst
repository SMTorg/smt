Rosenbrock function
===================

.. math ::
  \sum\limits_{i=1}^{nx-1}\left[(x_{i+1}-x_i^2)^2+(x_i-1)^2\right],\quad-2\leq x_i\leq 2,\quad\text{ for }i=1,\ldots,nx.

Usage
-----

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.problems import Rosenbrock
  
  ndim = 2
  problem = Rosenbrock(ndim=ndim)
  
  num = 100
  x = np.ones((num, ndim))
  x[:, 0] = np.linspace(-2, 2.0, num)
  x[:, 1] = 0.0
  y = problem(x)
  
  yd = np.empty((num, ndim))
  for i in range(ndim):
      yd[:, i] = problem(x, kx=i).flatten()
  
  print(y.shape)
  print(yd.shape)
  
  plt.plot(x[:, 0], y[:, 0])
  plt.xlabel("x")
  plt.ylabel("y")
  plt.show()
  
::

  (100, 1)
  (100, 2)
  
.. figure:: rosenbrock_Test_test_rosenbrock.png
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
  *  -  ndim
     -  1
     -  None
     -  ['int']
     -  
  *  -  return_complex
     -  False
     -  None
     -  ['bool']
     -  
  *  -  name
     -  Rosenbrock
     -  None
     -  ['str']
     -  
