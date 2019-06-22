Welded beam function
====================

.. math ::
  \sqrt{\frac{\tau'^2+\tau''^2+l\tau'\tau''}{\sqrt{0.25\left(l^2+(h+t)^2\right)}}},

where
:math:`\tau'=\frac{6000}{\sqrt{2}hl}, \quad\tau''=\frac{6000(14+0.5l)\sqrt{0.25\left(l^2+(h+t)^2\right)}}{2\left[0.707hl\left(\frac{l^2}{12}+0.25(h+t)^2\right)\right]},\quad \text{for}\quad h\in[0.125,1],\quad l,t\in[5,10].`

Usage
-----

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.problems import WeldedBeam
  
  ndim = 3
  problem = WeldedBeam(ndim=ndim)
  
  num = 100
  x = np.ones((num, ndim))
  for i in range(ndim):
      x[:, i] = 0.5 * (problem.xlimits[i, 0] + problem.xlimits[i, 1])
  x[:, 0] = np.linspace(5.0, 10.0, num)
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
  (100, 3)
  
.. figure:: weldedbeam_Test_test_welded_beam.png
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
     -  WeldedBeam
     -  None
     -  ['str']
     -  
  *  -  use_FD
     -  False
     -  None
     -  ['bool']
     -  
