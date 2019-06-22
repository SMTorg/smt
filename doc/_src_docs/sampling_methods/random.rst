Random sampling
===============

This class creates random samples from a uniform distribution over the design space.

Usage
-----

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.sampling_methods import Random
  
  xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
  sampling = Random(xlimits=xlimits)
  
  num = 50
  x = sampling(num)
  
  print(x.shape)
  
  plt.plot(x[:, 0], x[:, 1], "o")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.show()
  
::

  (50, 2)
  
.. figure:: random_Test_test_random.png
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
  *  -  xlimits
     -  None
     -  None
     -  ['ndarray']
     -  The interval of the domain in each dimension with shape nx x 2 (required)
