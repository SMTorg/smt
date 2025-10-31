1-D step-like data set
======================

.. code-block:: python

  import numpy as np
  
  
  def get_one_d_step():
      xt = np.array(
          [
              0.0000,
              0.4000,
              0.6000,
              0.7000,
              0.7500,
              0.7750,
              0.8000,
              0.8500,
              0.8750,
              0.9000,
              0.9250,
              0.9500,
              0.9750,
              1.0000,
              1.0250,
              1.0500,
              1.1000,
              1.2000,
              1.3000,
              1.4000,
              1.6000,
              1.8000,
              2.0000,
          ],
          dtype=np.float64,
      )
      yt = np.array(
          [
              0.0130,
              0.0130,
              0.0130,
              0.0130,
              0.0130,
              0.0130,
              0.0130,
              0.0132,
              0.0135,
              0.0140,
              0.0162,
              0.0230,
              0.0275,
              0.0310,
              0.0344,
              0.0366,
              0.0396,
              0.0410,
              0.0403,
              0.0390,
              0.0360,
              0.0350,
              0.0345,
          ],
          dtype=np.float64,
      )
  
      xlimits = np.array([[0.0, 2.0]])
  
      return xt, yt, xlimits
  
  
  def plot_one_d_step(xt, yt, limits, interp):
      import matplotlib
      import numpy as np
  
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
  
      num = 500
      x = np.linspace(0.0, 2.0, num)
      y = interp.predict_values(x)[:, 0]
  
      plt.plot(x, y)
      plt.plot(xt, yt, "o")
      plt.xlabel("x")
      plt.ylabel("y")
      plt.show()
  

RMTB
----

.. code-block:: python

  from smt.examples.one_D_step.one_D_step import get_one_d_step, plot_one_d_step
  from smt.surrogate_models import RMTB
  
  xt, yt, xlimits = get_one_d_step()
  
  interp = RMTB(
      num_ctrl_pts=100,
      xlimits=xlimits,
      nonlinear_maxiter=20,
      solver_tolerance=1e-16,
      energy_weight=1e-14,
      regularization_weight=0.0,
  )
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
           Computing dof2coeff - done. Time (sec):  0.0000000
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0000000
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0000000
        Pre-computing matrices - done. Time (sec):  0.0000000
        Solving for degrees of freedom ...
           Solving initial startup problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.032652876e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.127974095e-08 2.219243982e-13
              Solving for output 0 - done. Time (sec):  0.0060580
           Solving initial startup problem (n=100) - done. Time (sec):  0.0060580
           Solving nonlinear problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.548191750e-11 2.217751325e-13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.394278805e-11 2.190097980e-13
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.584836663e-10 1.413359358e-13
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.608388849e-10 1.074624094e-13
                 Iteration (num., iy, grad. norm, func.) :   3   0 1.062566066e-10 2.714188761e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 3.110030538e-11 1.186730380e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 8.637889286e-12 8.985747510e-15
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.113680400e-12 8.519465166e-15
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.080738024e-12 8.518276630e-15
                 Iteration (num., iy, grad. norm, func.) :   8   0 3.841507903e-13 8.471580148e-15
                 Iteration (num., iy, grad. norm, func.) :   9   0 3.112306577e-13 8.467274773e-15
                 Iteration (num., iy, grad. norm, func.) :  10   0 5.070566370e-14 8.454548297e-15
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.666762121e-14 8.453802707e-15
                 Iteration (num., iy, grad. norm, func.) :  12   0 1.727503879e-14 8.453801200e-15
                 Iteration (num., iy, grad. norm, func.) :  13   0 1.466105530e-14 8.453708354e-15
                 Iteration (num., iy, grad. norm, func.) :  14   0 9.493520089e-15 8.453377554e-15
                 Iteration (num., iy, grad. norm, func.) :  15   0 6.800282381e-15 8.453310106e-15
                 Iteration (num., iy, grad. norm, func.) :  16   0 8.753012817e-16 8.453274195e-15
                 Iteration (num., iy, grad. norm, func.) :  17   0 8.861540187e-16 8.453274132e-15
                 Iteration (num., iy, grad. norm, func.) :  18   0 5.330033187e-16 8.453273264e-15
                 Iteration (num., iy, grad. norm, func.) :  19   0 5.785118903e-16 8.453271091e-15
              Solving for output 0 - done. Time (sec):  0.0940828
           Solving nonlinear problem (n=100) - done. Time (sec):  0.0940828
        Solving for degrees of freedom - done. Time (sec):  0.1001408
     Training - done. Time (sec):  0.1001408
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: ex_1d_step.png
  :scale: 80 %
  :align: center

RMTC
----

.. code-block:: python

  from smt.examples.one_D_step.one_D_step import get_one_d_step, plot_one_d_step
  from smt.surrogate_models import RMTC
  
  xt, yt, xlimits = get_one_d_step()
  
  interp = RMTC(
      num_elements=40,
      xlimits=xlimits,
      nonlinear_maxiter=20,
      solver_tolerance=1e-16,
      energy_weight=1e-14,
      regularization_weight=0.0,
  )
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
           Computing dof2coeff - done. Time (sec):  0.0000000
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0000000
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0000000
        Pre-computing matrices - done. Time (sec):  0.0000000
        Solving for degrees of freedom ...
           Solving initial startup problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.470849329e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.807875749e-12 2.493686470e-14
              Solving for output 0 - done. Time (sec):  0.0000000
           Solving initial startup problem (n=82) - done. Time (sec):  0.0000000
           Solving nonlinear problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.484146522e-12 2.493686350e-14
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.032461792e-12 2.483319895e-14
                 Iteration (num., iy, grad. norm, func.) :   1   0 8.726294577e-11 2.394210072e-14
                 Iteration (num., iy, grad. norm, func.) :   2   0 6.860390512e-11 1.978091449e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 4.691798616e-11 1.537297203e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 9.922338291e-12 1.153328544e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 5.460856036e-12 1.130225803e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 8.530617619e-13 1.110676984e-14
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.870453869e-13 1.109190883e-14
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.151673802e-13 1.109065775e-14
                 Iteration (num., iy, grad. norm, func.) :   9   0 3.661383211e-14 1.108964365e-14
                 Iteration (num., iy, grad. norm, func.) :  10   0 9.092762497e-15 1.108943182e-14
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.449202696e-15 1.108940466e-14
                 Iteration (num., iy, grad. norm, func.) :  12   0 1.011249189e-16 1.108940343e-14
                 Iteration (num., iy, grad. norm, func.) :  13   0 1.154891849e-17 1.108940340e-14
              Solving for output 0 - done. Time (sec):  0.0649855
           Solving nonlinear problem (n=82) - done. Time (sec):  0.0649855
        Solving for degrees of freedom - done. Time (sec):  0.0649855
     Training - done. Time (sec):  0.0649855
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: ex_1d_step.png
  :scale: 80 %
  :align: center
