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
      import numpy as np
      import matplotlib
  
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

  from smt.surrogate_models import RMTB
  from smt.examples.one_D_step.one_D_step import get_one_d_step, plot_one_d_step
  
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
           Initializing Hessian - done. Time (sec):  0.0009968
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0009971
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0000000
        Pre-computing matrices - done. Time (sec):  0.0019939
        Solving for degrees of freedom ...
           Solving initial startup problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.032652876e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 8.879126107e-10 2.217748918e-13
              Solving for output 0 - done. Time (sec):  0.0049868
           Solving initial startup problem (n=100) - done. Time (sec):  0.0049868
           Solving nonlinear problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.552681635e-11 2.217739998e-13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.400336329e-11 2.190101681e-13
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.614334315e-10 1.420996036e-13
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.386385427e-10 9.957696091e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 9.965548477e-11 2.559472351e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 3.028292057e-11 1.206316456e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 8.399776244e-12 9.089294075e-15
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.074830115e-12 8.557170853e-15
                 Iteration (num., iy, grad. norm, func.) :   7   0 3.817538099e-13 8.464326104e-15
                 Iteration (num., iy, grad. norm, func.) :   8   0 2.920095323e-13 8.459776827e-15
                 Iteration (num., iy, grad. norm, func.) :   9   0 7.721236909e-14 8.454758697e-15
                 Iteration (num., iy, grad. norm, func.) :  10   0 2.417042968e-14 8.453876676e-15
                 Iteration (num., iy, grad. norm, func.) :  11   0 2.213769117e-14 8.453844066e-15
                 Iteration (num., iy, grad. norm, func.) :  12   0 7.168882034e-15 8.453415843e-15
                 Iteration (num., iy, grad. norm, func.) :  13   0 1.322758762e-15 8.453276708e-15
                 Iteration (num., iy, grad. norm, func.) :  14   0 1.086918935e-15 8.453275361e-15
                 Iteration (num., iy, grad. norm, func.) :  15   0 9.562581577e-16 8.453275241e-15
                 Iteration (num., iy, grad. norm, func.) :  16   0 8.144965006e-16 8.453273757e-15
                 Iteration (num., iy, grad. norm, func.) :  17   0 6.113930019e-16 8.453272411e-15
                 Iteration (num., iy, grad. norm, func.) :  18   0 1.911917523e-16 8.453270747e-15
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.931585657e-16 8.453270703e-15
              Solving for output 0 - done. Time (sec):  0.0987685
           Solving nonlinear problem (n=100) - done. Time (sec):  0.0987685
        Solving for degrees of freedom - done. Time (sec):  0.1037552
     Training - done. Time (sec):  0.1057491
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

  from smt.surrogate_models import RMTC
  from smt.examples.one_D_step.one_D_step import get_one_d_step, plot_one_d_step
  
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
           Computing dof2coeff - done. Time (sec):  0.0010138
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0019786
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0000000
        Pre-computing matrices - done. Time (sec):  0.0029924
        Solving for degrees of freedom ...
           Solving initial startup problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.470849329e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.031960886e-12 2.493686471e-14
              Solving for output 0 - done. Time (sec):  0.0060124
           Solving initial startup problem (n=82) - done. Time (sec):  0.0060124
           Solving nonlinear problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.484146480e-12 2.493686331e-14
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.032461435e-12 2.483319871e-14
                 Iteration (num., iy, grad. norm, func.) :   1   0 8.676384412e-11 2.388494401e-14
                 Iteration (num., iy, grad. norm, func.) :   2   0 6.755496606e-11 1.951826538e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 4.611359154e-11 1.521578957e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 9.560517650e-12 1.149409997e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 5.356117813e-12 1.129031598e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 8.365028192e-13 1.110518657e-14
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.740075688e-13 1.109155907e-14
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.554156778e-13 1.109124706e-14
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.497358635e-13 1.109091009e-14
                 Iteration (num., iy, grad. norm, func.) :  10   0 3.279226021e-14 1.108962650e-14
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.384472847e-14 1.108945872e-14
                 Iteration (num., iy, grad. norm, func.) :  12   0 2.898607110e-15 1.108940698e-14
                 Iteration (num., iy, grad. norm, func.) :  13   0 3.214675952e-16 1.108940346e-14
                 Iteration (num., iy, grad. norm, func.) :  14   0 1.039690103e-17 1.108940340e-14
              Solving for output 0 - done. Time (sec):  0.0817838
           Solving nonlinear problem (n=82) - done. Time (sec):  0.0817838
        Solving for degrees of freedom - done. Time (sec):  0.0877962
     Training - done. Time (sec):  0.0907886
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: ex_1d_step.png
  :scale: 80 %
  :align: center
