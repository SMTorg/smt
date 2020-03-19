Efficient Global Optimization (EGO)
===================================

Bayesian Optimization
---------------------

Bayesian optimization is defined by Jonas Mockus in [1]_ as an optimization technique 
based upon the minimization of the expected deviation from the extremum of the studied function. 

The objective function is treated as a black-box function. A Bayesian strategy sees the objective 
as a random function and places a prior over it. The prior captures our beliefs about the behavior 
of the function. After gathering the function evaluations, which are treated as data, the prior is 
updated to form the posterior distribution over the objective function. The posterior distribution, 
in turn, is used to construct an acquisition function (often also referred to as infill sampling 
criterion) that determines what the next query point should be.

One of the earliest bodies of work on Bayesian optimisation that we are aware 
of are [2]_ and [3]_. Kushner used Wiener processes for one-dimensional problems.
Kushner’s decision model was based on maximizing the probability of improvement, and included a 
parameter that controlled the trade-off between ‘more global’ and ‘more local’ optimization, in 
the same spirit as the Exploration/Exploitation trade-off.

Meanwhile, in the former Soviet Union, Mockus and colleagues developed a multidimensional 
Bayesian optimization method using linear combinations of Wiener fields, some of which was 
published in English in [1]_. This paper also describes an acquisition function that 
is based on myopic expected improvement of the posterior, which has been widely adopted in 
Bayesian optimization as the Expected Improvement function.

In 1998, Jones used Gaussian processes together with the expected improvement function to 
successfully perform derivative-free optimization and experimental design through an algorithm 
called  Efficient  Global  Optimization, or EGO.

EGO
---

In what follows, we describe the Efficient Global Optimization (EGO) algorithm, 
as published in [4]_.

Let :math:`F` be an expensive black-box function to be minimized. We sample :math:`F` at the 
different locations :math:`X = \{x_1, x_2,\ldots,x_n\}` yielding the responses 
:math:`Y = \{y_1, y_2,\ldots,y_n\}`. We build a Kriging model (also called Gaussian process) 
with a mean function :math:`\mu` and a variance function :math:`\sigma^{2}`.

The next step is to compute the criterion EI. To do this, let us denote:

.. math::
	\begin{equation}
	f_{min} = \min \{y_1, y_2,\ldots,y_n\}.
	\end{equation}

The Expected Improvement function (EI) can be expressed:

.. math::
	\begin{equation}	
	E[I(x)] = E[\max(f_{min}-Y, 0)]
	\end{equation}

where :math:`Y` is the random variable following the distribution :math:`\mathcal{N}(\mu(x), \sigma^{2}(x))`.
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

Return : :math:`f_{min}` \# This is the best known solution after :math:`n_{iter}` iterations.

More details can be found in [4]_.

Implementation Note
-------------------

Beside the Expected Improvement, the implementation here offers two other infill criteria:

* SBO (Surrogate Based Optimization): directly using the prediction of the surrogate model (:math:`\mu`)
* UCB (Upper Confidence bound): using the 99% confidence interval :math:`\mu -3 \times \sigma`


References
----------

.. [1] Mockus, J. (1975). On Bayesian methods for seeking the extremum. In Optimization Techniques IFIP Technical Conference (pp. 400-404). Springer, Berlin, Heidelberg.

.. [2] Kushner, H. J. (1962). A versatile stochastic model of a function of unknown and time varying form. Journal of Mathematical Analysis and Applications, 5(1), 150-167.

.. [3] Kushner, H. J. (1964). A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise. Journal of Basic Engineering, 86(1), 97-106.

.. [4] Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. Journal of Global optimization, 13(4), 455-492.

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
  from scipy.stats import norm
  
  def function_test_1d(x):
      # function xsinx
      import numpy as np
  
      x = np.reshape(x, (-1,))
      y = np.zeros(x.shape)
      y = (x - 3.5) * np.sin((x - 3.5) / (np.pi))
      return y.reshape((-1, 1))
  
  n_iter = 6
  xlimits = np.array([[0.0, 25.0]])
  xdoe = np.atleast_2d([0, 7, 25]).T
  n_doe = xdoe.size
  
  criterion = "EI"  #'EI' or 'SBO' or 'UCB'
  
  ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits)
  
  x_opt, y_opt, ind_best, x_data, y_data, x_doe, y_doe = ego.optimize(
      fun=function_test_1d
  )
  print("Minimum in x={:.1f} with f(x)={:.1f}".format(float(x_opt), float(y_opt)))
  
  x_plot = np.atleast_2d(np.linspace(0, 25, 100)).T
  y_plot = function_test_1d(x_plot)
  
  fig = plt.figure(figsize=[10, 10])
  for i in range(n_iter):
      k = n_doe + i
      x_data_k = x_data[0:k]
      y_data_k = y_data[0:k]
      ego.gpr.set_training_values(x_data_k, y_data_k)
      ego.gpr.train()
  
      y_gp_plot = ego.gpr.predict_values(x_plot)
      y_gp_plot_sd = np.sqrt(ego.gpr.predict_variances(x_plot))
      y_ei_plot = -ego.EI(x_plot, y_data_k)
  
      ax = fig.add_subplot((n_iter + 1) // 2, 2, i + 1)
      ax1 = ax.twinx()
      ei, = ax1.plot(x_plot, y_ei_plot, color="red")
  
      true_fun, = ax.plot(x_plot, y_plot)
      data, = ax.plot(
          x_data_k, y_data_k, linestyle="", marker="o", color="orange"
      )
      if i < n_iter - 1:
          opt, = ax.plot(
              x_data[k], y_data[k], linestyle="", marker="*", color="r"
          )
      gp, = ax.plot(x_plot, y_gp_plot, linestyle="--", color="g")
      sig_plus = y_gp_plot + 3 * y_gp_plot_sd
      sig_moins = y_gp_plot - 3 * y_gp_plot_sd
      un_gp = ax.fill_between(
          x_plot.T[0], sig_plus.T[0], sig_moins.T[0], alpha=0.3, color="g"
      )
      lines = [true_fun, data, gp, un_gp, opt, ei]
      fig.suptitle("EGO optimization of $f(x) = x \sin{x}$")
      fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.8)
      ax.set_title("iteration {}".format(i + 1))
      fig.legend(
          lines,
          [
              "f(x)=xsin(x)",
              "Given data points",
              "Kriging prediction",
              "Kriging 99% confidence interval",
              "Next point to evaluate",
              "Expected improvment function",
          ],
      )
  plt.show()
  
::

  Minimum in x=18.9 with f(x)=-15.1
  
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
     -  criterion for next evaluation point determination: Expected Improvement,             Surrogate-Based Optimization or Upper Confidence Bound
  *  -  n_iter
     -  None
     -  None
     -  ['int']
     -  Number of optimizer steps
  *  -  n_max_optim
     -  20
     -  None
     -  ['int']
     -  Maximum number of internal optimizations
  *  -  n_start
     -  20
     -  None
     -  ['int']
     -  Number of optimization start points
  *  -  n_doe
     -  None
     -  None
     -  ['int']
     -  Number of points of the initial LHS doe, only used if xdoe is not given
  *  -  xdoe
     -  None
     -  None
     -  ['ndarray']
     -  Initial doe inputs
  *  -  ydoe
     -  None
     -  None
     -  ['ndarray']
     -  Initial doe outputs
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
