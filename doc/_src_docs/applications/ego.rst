Efficient Global Optimization (EGO)
===================================

History
-------

Bayesian optimization is defined by Jonas Mockus in (Mockus, 1975) as an optimization technique 
based upon the minimization of the expected deviation from the extremum of the studied function. 

The objective function is treated as a black-box function. A Bayesian strategy sees the objective 
as a random function and places a prior over it. The prior captures our beliefs about the behavior 
of the function. After gathering the function evaluations, which are treated as data, the prior is 
updated to form the posterior distribution over the objective function. The posterior distribution, 
in turn, is used to construct an acquisition function (often also referred to as infill sampling 
criterion) that determines what the next query point should be.

One of the earliest bodies of work on Bayesian optimisation that we are aware 
of is (Kushner, 1962 ; Kushner, 1964). Kushner used Wiener processes for one-dimensional problems.
Kushner’s decision model was based on maximizing the probability of improvement, and included a 
parameter that controlled the trade-off between ‘more global’ and ‘more local’ optimization, in 
the same spirit as the Exploration/Exploitation trade-off.

Meanwhile, in the former Soviet Union, Mockus and colleagues developed a multidimensional 
Bayesian optimization method using linear combinations of Wiener fields, some of which was 
published in English in (Mockus, 1975). This paper also describes an acquisition function that 
is based on myopic expected improvement of the posterior, which has been widely adopted in 
Bayesian optimization as the Expected Improvement function.

In 1998, Jones used Gaussian processes together with the expected improvement function to 
successfully perform derivative-free optimization and experimental design through an algorithm 
called  Efficient  Global  Optimization, or EGO (Jones, 1998).

EGO
---

In what follows, we describe the Efficient Global Optimization (EGO) algorithm, 
as published in (Jones, 1998).

Let :math:`F` be an expensive black-box function to be minimized. We sample :math:`F` at the 
different locations :math:`X = \{x_1, x_2,\ldots,x_n\}$` yielding the responses 
:math:`Y = \{y_1, y_2,\ldots,y_n\}`. We build a Kriging model (also called Gaussian process) 
with a mean function :math:`\mu` and a variance function :math:`\sigma^{2}`.

The next step is to compute the criterion EI. To do this, let us denote:

.. math::
	\begin{equation}
	f_{min} = \min \{y_1, y_2,\ldots,y_n\}.
	\end{equation}

The Expected Improvement funtion (EI) can be expressed:

.. math::
	\begin{equation}	
	E[I(x)] = E[\max(f_{min}-Y, 0)],
	\end{equation}

where :math:`Y` is the random variable following the distribution
 :math:`\mathcal{N}(\mu(x), \sigma^{2}(x))`.
By expressing the right-hand side of EI expression as an integral, and applying some tedious 
integration by parts, one can express the expected improvement in closed form: 

.. math::
  \begin{equation}	
  E[I(x)] = (f_{min} - \mu(x))\Phi\left(\frac{f_{min} - \mu(x)}{\sigma(x)}\right) + \sigma(x) \phi\left(\frac{f_{min} - \mu(x)}{\sigma(x)}\right)
  \end{equation}

where :math:`\Phi(\cdot)` and :math:`\phi(\cdot)` are respectively the cumulative and probability 
density functions of :math:`\mathcal{N}(0,1)`.

Next, we determine our next sampling point as :

.. math::
	\begin{equation}
	x_{n+1} = \arg \max_{x} \left(E[I(x)]\right)
	\end{equation}

We then test the response :math:`y_{n+1}` of our black-box function :math:`F` at :math:`x_{n+1}`, 
rebuild the model taking into account the new information gained, and research 
the point of maximum expected improvement again.

We summarize here the EGO algorithm:

EGO(F, :math:`n_{iter}`) \# Find the best minimum of :math:`\operatorname{F}` 
in :math:`n_{iter}` iterations  

For (:math:`i=0:n_{iter}`)  

* :math:`mod = {model}(X, Y)`  \# surrogate model based on sample vectors :math:`X` and :math:`Y`  
* :math:`f_{min} = \min Y`  
* :math:`x_{i+1} = \arg \max {EI}(mod, f_{min})` \# choose :math:`x` that maximizes EI  
* :math:`y_{i+1} = {F}(x_{i+1})` \# Probe the function at most promising point :math:`x_{i+1}`  
* :math:`X = [X,x_{i+1}]`  
* :math:`Y = [Y,y_{i+1}]`   
* :math:`i = i+1`  

:math:`f_{min} = \min Y`  
Return : :math:`f_{min}` \# This is the best known solution after :math:`n_{iter}` iterations


.. math ::
	\begin{equation}\label{e:globalMOE}
	\hat{y}({\bf x})=\sum_{i=1}^{K} \mathbb{P}(\kappa=i|X={\bf x}) \hat{y_i}({\bf x})
	\end{equation}

which is the classical probability expression of mixture of experts.

In this equation, :math:`K` is the number of Gaussian components, :math:`\mathbb{P}(\kappa=i|X= {\bf x})`, denoted by gating network,  is the probability to lie in cluster :math:`i` knowing that :math:`X = {\bf x}` and :math:`\hat{y_i}` is the local expert built on cluster :math:`i`.

This equation leads to two different approximation models depending on the computation of :math:`\mathbb{P}(\kappa=i|X={\bf x})`. 

	* When choosing the Gaussian laws to compute this quantity, the equation leads to a *smooth model* that smoothly recombine different local experts.
	* If :math:`\mathbb{P}(\kappa=i|X= {\bf x})` is computed as characteristic functions of clusters (being equal to 0 or 1) this leads to a *discontinuous approximation model*.

More details can be found in [1]_.

References
----------

.. [1] Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. Journal of Global optimization, 13(4), 455-492.

Usage
-----

.. code-block:: python

  import numpy as np
  import six
  from smt.applications import EGO
  from smt.sampling_methods import FullFactorial
  
  import sklearn
  import matplotlib.pyplot as plt
  from matplotlib import colors
  from mpl_toolkits.mplot3d import Axes3D
  
  def function_test_1d(x):
      # function xsinx
      import numpy as np
  
      x = np.reshape(x, (-1,))
      y = np.zeros(x.shape)
      y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
      return y.reshape((-1, 1))
  
  ndim = 1
  niter = 15
  xlimits = np.array([[0.0, 25.0]])
  
  criterion = "UCB"  #'EI' or 'SBO' or 'UCB'
  
  ego = EGO(niter=niter, criterion=criterion, ndoe=3, xlimits=xlimits)
  
  x_opt, y_opt, _, _, _, _, _ = ego.optimize(fun=function_test_1d)
  print(x_opt, y_opt)
  
::

  [18.93526158] [-15.12510323]
  
.. figure:: ego_TestEGO_run_ego_example.png
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
  *  -  fun
     -  None
     -  None
     -  ['function']
     -  Function to minimize
  *  -  criterion
     -  EI
     -  ['EI', 'SBO', 'UCB']
     -  ['str']
     -  criterion for next evaluaition point
  *  -  niter
     -  None
     -  None
     -  ['int']
     -  Number of iterations
  *  -  nmax_optim
     -  20
     -  None
     -  ['int']
     -  Maximum number of internal optimizations
  *  -  nstart
     -  20
     -  None
     -  ['int']
     -  Number of start
  *  -  ndoe
     -  None
     -  None
     -  ['int']
     -  Number of points of the initial doe
  *  -  xdoe
     -  None
     -  None
     -  ['ndarray']
     -  Initial doe inputs
  *  -  xlimits
     -  None
     -  None
     -  ['ndarray']
     -  Bounds of function fun inputs
  *  -  verbose
     -  False
     -  None
     -  ['bool']
     -  Print computation information
