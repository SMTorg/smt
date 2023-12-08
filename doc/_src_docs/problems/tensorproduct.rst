Tensor-product function
=======================

.. rubric :: cos

.. math ::
  \prod\limits_{i=1}^{nx}\cos(a\pi x_i),\quad-1\leq x_i\leq 1,\quad\text{ for }i=1,\ldots,nx.

.. rubric :: exp

.. math ::
  \prod\limits_{i=1}^{nx}\exp(x_i),\quad-1\leq x_i\leq 1,\quad\text{ for }i=1,\ldots,nx.

.. rubric :: tanh

.. math ::
  \prod\limits_{i=1}^{nx}\tanh(x_i),\quad-1\leq x_i\leq 1,\quad\text{ for }i=1,\ldots,nx.

.. rubric :: gaussian

.. math ::
  \prod\limits_{i=1}^{nx}\exp(-2 x_i^2),\quad-1\leq x_i\leq 1,\quad\text{ for }i=1,\ldots,nx.

Usage
-----

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.problems import TensorProduct
  
  ndim = 2
  problem = TensorProduct(ndim=ndim, func="cos")
  
  num = 100
  x = np.ones((num, ndim))
  x[:, 0] = np.linspace(-1, 1.0, num)
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
  
.. figure:: tensorproduct_Test_test_tensor_product.png
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
     -  TP
     -  None
     -  ['str']
     -  
  *  -  func
     -  None
     -  ['cos', 'exp', 'tanh', 'gaussian']
     -  None
     -  
  *  -  width
     -  1.0
     -  None
     -  ['float', 'int']
     -  
