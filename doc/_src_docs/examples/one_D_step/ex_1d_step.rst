1-D step-like data set
======================

.. code-block:: python

  import numpy as np
  
  
  def get_one_d_step():
      xt = np.array([
          0.0000,    0.4000,    0.6000,    0.7000,    0.7500,
          0.7750,    0.8000,    0.8500,    0.8750,    0.9000,
          0.9250,    0.9500,    0.9750,    1.0000,    1.0250,
          1.0500,    1.1000,    1.2000,    1.3000,    1.4000,
          1.6000,    1.8000,    2.0000,
      ], dtype=np.float64)
      yt = np.array([
          0.0130,     0.0130,     0.0130,     0.0130,   0.0130,
          0.0130,     0.0130,     0.0132,     0.0135,   0.0140,
          0.0162,     0.0230,     0.0275,     0.0310,   0.0344,
          0.0366,     0.0396,     0.0410,     0.0403,   0.0390,
          0.0360,     0.0350,     0.0345,
      ], dtype=np.float64)
  
      xlimits = np.array([[0.0, 2.0]])
  
      return xt, yt, xlimits
  
  
  def plot_one_d_step(xt, yt, limits, interp):
      import numpy as np
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
  
      num = 500
      x = np.linspace(0., 2., num)
      y = interp.predict_values(x)[:, 0]
  
      plt.plot(x, y)
      plt.plot(xt, yt, 'o')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.show()
  

RMTB
----

.. code-block:: python

  from smt.surrogate_models import RMTB
  from smt.examples.one_D_step.one_D_step import get_one_d_step, plot_one_d_step
  
  xt, yt, xlimits = get_one_d_step()
  
  interp = RMTB(num_ctrl_pts=100, xlimits=xlimits, nonlinear_maxiter=20,
      solver_tolerance=1e-16, energy_weight=1e-14, regularization_weight=0.)
  interp.set_training_values(xt, yt)
  interp.train()
  
  plot_one_d_step(xt, yt, xlimits, interp)
  
::

  ___________________________________________________________________________
     
                                     RMTB
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 23
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Pre-computing matrices ...
           Computing dof2coeff ...
           Computing dof2coeff - done. Time (sec):  0.0000021
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0004480
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0017099
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0005701
        Pre-computing matrices - done. Time (sec):  0.0028081
        Solving for degrees of freedom ...
           Solving initial startup problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.032652876e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.825713981e-09 2.217815952e-13
              Solving for output 0 - done. Time (sec):  0.0111849
           Solving initial startup problem (n=100) - done. Time (sec):  0.0112500
           Solving nonlinear problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.552478064e-11 2.217739692e-13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.398604938e-11 2.190065593e-13
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.568460900e-10 1.408376384e-13
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.695778954e-10 1.123176246e-13
                 Iteration (num., iy, grad. norm, func.) :   3   0 1.093471806e-10 2.876640533e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 6.776407500e-11 2.001901997e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 5.303098517e-11 1.710293788e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.515151561e-11 9.974345459e-15
                 Iteration (num., iy, grad. norm, func.) :   7   0 4.058068330e-12 8.686676540e-15
                 Iteration (num., iy, grad. norm, func.) :   8   0 9.167765353e-13 8.480951347e-15
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.973788463e-13 8.459739902e-15
                 Iteration (num., iy, grad. norm, func.) :  10   0 5.417639479e-14 8.454609964e-15
                 Iteration (num., iy, grad. norm, func.) :  11   0 3.262736039e-14 8.453956388e-15
                 Iteration (num., iy, grad. norm, func.) :  12   0 9.376713988e-15 8.453393992e-15
                 Iteration (num., iy, grad. norm, func.) :  13   0 5.567537262e-15 8.453322024e-15
                 Iteration (num., iy, grad. norm, func.) :  14   0 3.915800764e-15 8.453293857e-15
                 Iteration (num., iy, grad. norm, func.) :  15   0 3.160681038e-15 8.453291334e-15
                 Iteration (num., iy, grad. norm, func.) :  16   0 7.900787119e-16 8.453275434e-15
                 Iteration (num., iy, grad. norm, func.) :  17   0 3.493093271e-16 8.453271517e-15
                 Iteration (num., iy, grad. norm, func.) :  18   0 3.493095441e-16 8.453271517e-15
                 Iteration (num., iy, grad. norm, func.) :  19   0 3.493099781e-16 8.453271517e-15
              Solving for output 0 - done. Time (sec):  0.2039640
           Solving nonlinear problem (n=100) - done. Time (sec):  0.2040122
        Solving for degrees of freedom - done. Time (sec):  0.2153280
     Training - done. Time (sec):  0.2186990
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005391
     
     Prediction time/pt. (sec) :  0.0000011
     
  
.. figure:: ex_1d_step.png
  :scale: 80 %
  :align: center

RMTC
----

.. code-block:: python

  from smt.surrogate_models import RMTC
  from smt.examples.one_D_step.one_D_step import get_one_d_step, plot_one_d_step
  
  xt, yt, xlimits = get_one_d_step()
  
  interp = RMTC(num_elements=40, xlimits=xlimits, nonlinear_maxiter=20,
      solver_tolerance=1e-16, energy_weight=1e-14, regularization_weight=0.)
  interp.set_training_values(xt, yt)
  interp.train()
  
  plot_one_d_step(xt, yt, xlimits, interp)
  
::

  ___________________________________________________________________________
     
                                     RMTC
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 23
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Pre-computing matrices ...
           Computing dof2coeff ...
           Computing dof2coeff - done. Time (sec):  0.0012641
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0005240
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0021732
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0009310
        Pre-computing matrices - done. Time (sec):  0.0049980
        Solving for degrees of freedom ...
           Solving initial startup problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.470849329e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.365376360e-10 2.493682992e-14
              Solving for output 0 - done. Time (sec):  0.0139589
           Solving initial startup problem (n=82) - done. Time (sec):  0.0140400
           Solving nonlinear problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.484138768e-12 2.493679513e-14
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.032452097e-12 2.483313104e-14
                 Iteration (num., iy, grad. norm, func.) :   1   0 8.649861558e-11 2.385471045e-14
                 Iteration (num., iy, grad. norm, func.) :   2   0 8.463306052e-11 2.108728233e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 2.476214424e-11 1.285665016e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 1.287993204e-11 1.181281675e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 3.607090697e-12 1.120417686e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.060038504e-12 1.111157211e-14
                 Iteration (num., iy, grad. norm, func.) :   7   0 6.362787472e-13 1.110179625e-14
                 Iteration (num., iy, grad. norm, func.) :   8   0 4.933044810e-13 1.109821782e-14
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.445626803e-13 1.109102057e-14
                 Iteration (num., iy, grad. norm, func.) :  10   0 4.130423613e-14 1.108966863e-14
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.095957805e-14 1.108943666e-14
                 Iteration (num., iy, grad. norm, func.) :  12   0 2.266037285e-15 1.108940544e-14
                 Iteration (num., iy, grad. norm, func.) :  13   0 2.187923327e-16 1.108940342e-14
                 Iteration (num., iy, grad. norm, func.) :  14   0 3.171473200e-18 1.108940339e-14
              Solving for output 0 - done. Time (sec):  0.2069931
           Solving nonlinear problem (n=82) - done. Time (sec):  0.2070560
        Solving for degrees of freedom - done. Time (sec):  0.2211740
     Training - done. Time (sec):  0.2268560
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006320
     
     Prediction time/pt. (sec) :  0.0000013
     
  
.. figure:: ex_1d_step.png
  :scale: 80 %
  :align: center
