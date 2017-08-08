Random sampling
===============

This class creates random samples from a uniform distribution over the design space.

Usage
-----

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.sampling import Random
  
  xlimits = np.array([
      [0., 4.],
      [0., 3.],
  ])
  sampling = Random(xlimits=xlimits)
  
  num = 50
  x = sampling(num)
  
  print(x.shape)
  
  plt.plot(x[:, 0], x[:, 1], 'o')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()
  
::

  (50, 2)
  
.. figure:: random.png
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
  *  -  xlimits
     -  None
     -  None
     -  ['ndarray']
     -  
