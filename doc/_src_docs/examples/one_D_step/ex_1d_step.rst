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
           Computing energy terms - done. Time (sec):  0.0156224
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0000000
        Pre-computing matrices - done. Time (sec):  0.0156224
        Solving for degrees of freedom ...
           Solving initial startup problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.032652876e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 6.786680863e-09 2.218151080e-13
              Solving for output 0 - done. Time (sec):  0.0000000
           Solving initial startup problem (n=100) - done. Time (sec):  0.0000000
           Solving nonlinear problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.551467973e-11 2.217740430e-13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.401077818e-11 2.190113743e-13
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.566246092e-10 1.398094006e-13
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.187142258e-10 9.327244974e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 9.374199098e-11 2.441319143e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 7.181676598e-11 1.894335409e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 2.109617230e-11 1.047182645e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 8.050659814e-12 9.103083561e-15
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.350748228e-12 8.576088440e-15
                 Iteration (num., iy, grad. norm, func.) :   8   0 6.944357725e-13 8.476438536e-15
                 Iteration (num., iy, grad. norm, func.) :   9   0 2.304594377e-13 8.459868871e-15
                 Iteration (num., iy, grad. norm, func.) :  10   0 2.613552317e-13 8.459867906e-15
                 Iteration (num., iy, grad. norm, func.) :  11   0 7.561345920e-14 8.454412587e-15
                 Iteration (num., iy, grad. norm, func.) :  12   0 2.392678153e-14 8.453533882e-15
                 Iteration (num., iy, grad. norm, func.) :  13   0 6.369300553e-15 8.453334168e-15
                 Iteration (num., iy, grad. norm, func.) :  14   0 1.376985434e-15 8.453277655e-15
                 Iteration (num., iy, grad. norm, func.) :  15   0 9.337089436e-16 8.453273240e-15
                 Iteration (num., iy, grad. norm, func.) :  16   0 7.890066819e-16 8.453272942e-15
                 Iteration (num., iy, grad. norm, func.) :  17   0 6.756594233e-16 8.453271528e-15
                 Iteration (num., iy, grad. norm, func.) :  18   0 2.143825648e-16 8.453270718e-15
                 Iteration (num., iy, grad. norm, func.) :  19   0 7.115441072e-17 8.453270419e-15
              Solving for output 0 - done. Time (sec):  0.1009324
           Solving nonlinear problem (n=100) - done. Time (sec):  0.1009324
        Solving for degrees of freedom - done. Time (sec):  0.1009324
     Training - done. Time (sec):  0.1165547
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
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.672771647e-10 2.493685484e-14
              Solving for output 0 - done. Time (sec):  0.0000000
           Solving initial startup problem (n=82) - done. Time (sec):  0.0000000
           Solving nonlinear problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.484144240e-12 2.493684372e-14
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.032459200e-12 2.483317928e-14
                 Iteration (num., iy, grad. norm, func.) :   1   0 8.718228593e-11 2.392796282e-14
                 Iteration (num., iy, grad. norm, func.) :   2   0 4.084733341e-11 1.653726244e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.968335331e-11 1.635461268e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 1.075192906e-11 1.163951275e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 2.996703771e-12 1.114623030e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 4.613969124e-13 1.109347169e-14
                 Iteration (num., iy, grad. norm, func.) :   7   0 9.192134677e-14 1.109036961e-14
                 Iteration (num., iy, grad. norm, func.) :   8   0 5.561355733e-14 1.108986678e-14
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.518497445e-14 1.108946807e-14
                 Iteration (num., iy, grad. norm, func.) :  10   0 3.525949088e-15 1.108940914e-14
                 Iteration (num., iy, grad. norm, func.) :  11   0 4.977597964e-16 1.108940364e-14
                 Iteration (num., iy, grad. norm, func.) :  12   0 4.328767946e-17 1.108940341e-14
              Solving for output 0 - done. Time (sec):  0.0691741
           Solving nonlinear problem (n=82) - done. Time (sec):  0.0691741
        Solving for degrees of freedom - done. Time (sec):  0.0691741
     Training - done. Time (sec):  0.0691741
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: ex_1d_step.png
  :scale: 80 %
  :align: center
