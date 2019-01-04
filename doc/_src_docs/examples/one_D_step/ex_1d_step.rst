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
           Computing dof2coeff - done. Time (sec):  0.0000029
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0004518
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0016558
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0005691
        Pre-computing matrices - done. Time (sec):  0.0027540
        Solving for degrees of freedom ...
           Solving initial startup problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.032652876e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.643333520e-09 2.217757928e-13
              Solving for output 0 - done. Time (sec):  0.0125730
           Solving initial startup problem (n=100) - done. Time (sec):  0.0128310
           Solving nonlinear problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.552653928e-11 2.217739854e-13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.399474164e-11 2.190091350e-13
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.607781546e-10 1.418763647e-13
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.543191897e-10 1.039559166e-13
                 Iteration (num., iy, grad. norm, func.) :   3   0 1.043058603e-10 2.645222339e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 3.026732601e-11 1.169063841e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 2.851994842e-11 1.150853781e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 7.891991175e-12 8.937584128e-15
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.984599657e-12 8.517082789e-15
                 Iteration (num., iy, grad. norm, func.) :   8   0 3.543987866e-13 8.459539371e-15
                 Iteration (num., iy, grad. norm, func.) :   9   0 3.433971389e-13 8.459215597e-15
                 Iteration (num., iy, grad. norm, func.) :  10   0 5.648581154e-14 8.454064551e-15
                 Iteration (num., iy, grad. norm, func.) :  11   0 2.299879661e-14 8.453515634e-15
                 Iteration (num., iy, grad. norm, func.) :  12   0 1.458231015e-14 8.453389938e-15
                 Iteration (num., iy, grad. norm, func.) :  13   0 2.217840933e-15 8.453309951e-15
                 Iteration (num., iy, grad. norm, func.) :  14   0 1.432812572e-15 8.453282436e-15
                 Iteration (num., iy, grad. norm, func.) :  15   0 2.141547677e-16 8.453270723e-15
                 Iteration (num., iy, grad. norm, func.) :  16   0 3.002938753e-16 8.453270690e-15
                 Iteration (num., iy, grad. norm, func.) :  17   0 2.820200543e-16 8.453270650e-15
                 Iteration (num., iy, grad. norm, func.) :  18   0 2.820202242e-16 8.453270650e-15
                 Iteration (num., iy, grad. norm, func.) :  19   0 2.820202455e-16 8.453270650e-15
              Solving for output 0 - done. Time (sec):  0.2801340
           Solving nonlinear problem (n=100) - done. Time (sec):  0.2802680
        Solving for degrees of freedom - done. Time (sec):  0.2931740
     Training - done. Time (sec):  0.2964010
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005271
     
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
           Computing dof2coeff - done. Time (sec):  0.0009160
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0003619
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0013752
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0007749
        Pre-computing matrices - done. Time (sec):  0.0035551
        Solving for degrees of freedom ...
           Solving initial startup problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.470849329e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.424635858e-09 2.493611941e-14
              Solving for output 0 - done. Time (sec):  0.0117810
           Solving initial startup problem (n=82) - done. Time (sec):  0.0118542
           Solving nonlinear problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.483974416e-12 2.493536699e-14
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.032504429e-12 2.483174006e-14
                 Iteration (num., iy, grad. norm, func.) :   1   0 8.684188511e-11 2.388973220e-14
                 Iteration (num., iy, grad. norm, func.) :   2   0 4.263422780e-11 1.666458893e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.987218819e-11 1.602159952e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.264023266e-11 1.327312190e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 2.465876892e-11 1.260042986e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 6.948092995e-12 1.134123925e-14
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.249055862e-12 1.115103431e-14
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.298397423e-12 1.111991758e-14
                 Iteration (num., iy, grad. norm, func.) :   9   0 3.432355496e-13 1.109523167e-14
                 Iteration (num., iy, grad. norm, func.) :  10   0 8.342707724e-14 1.109033612e-14
                 Iteration (num., iy, grad. norm, func.) :  11   0 2.312306903e-14 1.108955041e-14
                 Iteration (num., iy, grad. norm, func.) :  12   0 5.914379636e-15 1.108942138e-14
                 Iteration (num., iy, grad. norm, func.) :  13   0 1.169929511e-15 1.108940476e-14
                 Iteration (num., iy, grad. norm, func.) :  14   0 1.862796999e-16 1.108940350e-14
                 Iteration (num., iy, grad. norm, func.) :  15   0 1.877718181e-16 1.108940345e-14
                 Iteration (num., iy, grad. norm, func.) :  16   0 1.877718817e-16 1.108940345e-14
                 Iteration (num., iy, grad. norm, func.) :  17   0 1.877739157e-16 1.108940345e-14
                 Iteration (num., iy, grad. norm, func.) :  18   0 1.877739792e-16 1.108940345e-14
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.877739797e-16 1.108940345e-14
              Solving for output 0 - done. Time (sec):  0.2394340
           Solving nonlinear problem (n=82) - done. Time (sec):  0.2395000
        Solving for degrees of freedom - done. Time (sec):  0.2514260
     Training - done. Time (sec):  0.2555780
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005901
     
     Prediction time/pt. (sec) :  0.0000012
     
  
.. figure:: ex_1d_step.png
  :scale: 80 %
  :align: center
