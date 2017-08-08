Robot arm function
==================

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.problems import RobotArm
  
  ndim = 2
  problem = RobotArm(ndim=ndim)
  
  num = 100
  x = np.ones((num, ndim))
  x[:, 0] = np.linspace(0., 1., num)
  x[:, 1] = np.pi
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
  
.. figure:: robotarm.png
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
     -  2
     -  None
     -  ['int']
     -  
  *  -  return_complex
     -  False
     -  None
     -  ['bool']
     -  
  *  -  name
     -  RobotArm
     -  None
     -  ['str']
     -  
