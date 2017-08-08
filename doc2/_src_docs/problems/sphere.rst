Sphere function
===============

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.problems import Sphere
  
  ndim = 2
  problem = Sphere(ndim=ndim)
  
  num = 100
  x = np.ones((num, ndim))
  x[:, 0] = np.linspace(-10, 10., num)
  x[:, 1] = 0.
  y = problem(x)
  
  yd = np.empty((num, ndim))
  for i in range(ndim):
      yd[:, i] = problem(x, kx=i).flatten()
  
  print(y.shape)
  print(yd.shape)
  
  plt.plot(x[:, 0], y[:, 0])
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()
  
::

  (100, 1)
  (100, 2)
  
.. figure:: sphere.png
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
     -  Sphere
     -  None
     -  ['str']
     -  
