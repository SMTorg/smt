Boeing 777 engine data set
==========================

.. code-block:: python

  import numpy as np
  import os
  
  
  def get_b777_engine():
      this_dir = os.path.split(__file__)[0]
  
      nt = 12 * 11 * 8
      xt = np.loadtxt(os.path.join(this_dir, "b777_engine_inputs.dat")).reshape((nt, 3))
      yt = np.loadtxt(os.path.join(this_dir, "b777_engine_outputs.dat")).reshape((nt, 2))
      dyt_dxt = np.loadtxt(os.path.join(this_dir, "b777_engine_derivs.dat")).reshape(
          (nt, 2, 3)
      )
  
      xlimits = np.array([[0, 0.9], [0, 15], [0, 1.0]])
  
      return xt, yt, dyt_dxt, xlimits
  
  
  def plot_b777_engine(xt, yt, limits, interp):
      import numpy as np
      import matplotlib
  
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
  
      val_M = np.array(
          [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
      )  # 12
      val_h = np.array(
          [0.0, 0.6096, 1.524, 3.048, 4.572, 6.096, 7.62, 9.144, 10.668, 11.8872, 13.1064]
      )  # 11
      val_t = np.array([0.05, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 1.0])  # 8
  
      def get_pts(xt, yt, iy, ind_M=None, ind_h=None, ind_t=None):
          eps = 1e-5
  
          if ind_M is not None:
              M = val_M[ind_M]
              keep = abs(xt[:, 0] - M) < eps
              xt = xt[keep, :]
              yt = yt[keep, :]
          if ind_h is not None:
              h = val_h[ind_h]
              keep = abs(xt[:, 1] - h) < eps
              xt = xt[keep, :]
              yt = yt[keep, :]
          if ind_t is not None:
              t = val_t[ind_t]
              keep = abs(xt[:, 2] - t) < eps
              xt = xt[keep, :]
              yt = yt[keep, :]
  
          if ind_M is None:
              data = xt[:, 0], yt[:, iy]
          elif ind_h is None:
              data = xt[:, 1], yt[:, iy]
          elif ind_t is None:
              data = xt[:, 2], yt[:, iy]
  
          if iy == 0:
              data = data[0], data[1] / 1e6
          elif iy == 1:
              data = data[0], data[1] / 1e-4
  
          return data
  
      num = 100
      x = np.zeros((num, 3))
      lins_M = np.linspace(0.0, 0.9, num)
      lins_h = np.linspace(0.0, 13.1064, num)
      lins_t = np.linspace(0.05, 1.0, num)
  
      def get_x(ind_M=None, ind_h=None, ind_t=None):
          x = np.zeros((num, 3))
          x[:, 0] = lins_M
          x[:, 1] = lins_h
          x[:, 2] = lins_t
          if ind_M:
              x[:, 0] = val_M[ind_M]
          if ind_h:
              x[:, 1] = val_h[ind_h]
          if ind_t:
              x[:, 2] = val_t[ind_t]
          return x
  
      nrow = 6
      ncol = 2
  
      ind_M_1 = -2
      ind_M_2 = -5
  
      ind_t_1 = 1
      ind_t_2 = -1
  
      plt.close()
      plt.figure(figsize=(15, 25))
      plt.subplots_adjust(hspace=0.5)
  
      # --------------------
  
      plt.subplot(nrow, ncol, 1)
      plt.title("M={}".format(val_M[ind_M_1]))
      plt.xlabel("throttle")
      plt.ylabel("thrust (x 1e6 N)")
  
      plt.subplot(nrow, ncol, 2)
      plt.title("M={}".format(val_M[ind_M_1]))
      plt.xlabel("throttle")
      plt.ylabel("SFC (x 1e-3 N/N/s)")
  
      plt.subplot(nrow, ncol, 3)
      plt.title("M={}".format(val_M[ind_M_2]))
      plt.xlabel("throttle")
      plt.ylabel("thrust (x 1e6 N)")
  
      plt.subplot(nrow, ncol, 4)
      plt.title("M={}".format(val_M[ind_M_2]))
      plt.xlabel("throttle")
      plt.ylabel("SFC (x 1e-3 N/N/s)")
  
      # --------------------
  
      plt.subplot(nrow, ncol, 5)
      plt.title("throttle={}".format(val_t[ind_t_1]))
      plt.xlabel("altitude (km)")
      plt.ylabel("thrust (x 1e6 N)")
  
      plt.subplot(nrow, ncol, 6)
      plt.title("throttle={}".format(val_t[ind_t_1]))
      plt.xlabel("altitude (km)")
      plt.ylabel("SFC (x 1e-3 N/N/s)")
  
      plt.subplot(nrow, ncol, 7)
      plt.title("throttle={}".format(val_t[ind_t_2]))
      plt.xlabel("altitude (km)")
      plt.ylabel("thrust (x 1e6 N)")
  
      plt.subplot(nrow, ncol, 8)
      plt.title("throttle={}".format(val_t[ind_t_2]))
      plt.xlabel("altitude (km)")
      plt.ylabel("SFC (x 1e-3 N/N/s)")
  
      # --------------------
  
      plt.subplot(nrow, ncol, 9)
      plt.title("throttle={}".format(val_t[ind_t_1]))
      plt.xlabel("Mach number")
      plt.ylabel("thrust (x 1e6 N)")
  
      plt.subplot(nrow, ncol, 10)
      plt.title("throttle={}".format(val_t[ind_t_1]))
      plt.xlabel("Mach number")
      plt.ylabel("SFC (x 1e-3 N/N/s)")
  
      plt.subplot(nrow, ncol, 11)
      plt.title("throttle={}".format(val_t[ind_t_2]))
      plt.xlabel("Mach number")
      plt.ylabel("thrust (x 1e6 N)")
  
      plt.subplot(nrow, ncol, 12)
      plt.title("throttle={}".format(val_t[ind_t_2]))
      plt.xlabel("Mach number")
      plt.ylabel("SFC (x 1e-3 N/N/s)")
  
      ind_h_list = [0, 4, 7, 10]
      ind_h_list = [4, 7, 10]
  
      ind_M_list = [0, 3, 6, 11]
      ind_M_list = [3, 6, 11]
  
      colors = ["b", "r", "g", "c", "m"]
  
      # -----------------------------------------------------------------------------
  
      # Throttle slices
      for k, ind_h in enumerate(ind_h_list):
          ind_M = ind_M_1
          x = get_x(ind_M=ind_M, ind_h=ind_h)
          y = interp.predict_values(x)
  
          plt.subplot(nrow, ncol, 1)
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_h=ind_h)
          plt.plot(xt_, yt_, "o" + colors[k])
          plt.plot(lins_t, y[:, 0] / 1e6, colors[k])
          plt.subplot(nrow, ncol, 2)
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_h=ind_h)
          plt.plot(xt_, yt_, "o" + colors[k])
          plt.plot(lins_t, y[:, 1] / 1e-4, colors[k])
  
          ind_M = ind_M_2
          x = get_x(ind_M=ind_M, ind_h=ind_h)
          y = interp.predict_values(x)
  
          plt.subplot(nrow, ncol, 3)
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_h=ind_h)
          plt.plot(xt_, yt_, "o" + colors[k])
          plt.plot(lins_t, y[:, 0] / 1e6, colors[k])
          plt.subplot(nrow, ncol, 4)
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_h=ind_h)
          plt.plot(xt_, yt_, "o" + colors[k])
          plt.plot(lins_t, y[:, 1] / 1e-4, colors[k])
  
      # -----------------------------------------------------------------------------
  
      # Altitude slices
      for k, ind_M in enumerate(ind_M_list):
          ind_t = ind_t_1
          x = get_x(ind_M=ind_M, ind_t=ind_t)
          y = interp.predict_values(x)
  
          plt.subplot(nrow, ncol, 5)
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_t=ind_t)
          plt.plot(xt_, yt_, "o" + colors[k])
          plt.plot(lins_h, y[:, 0] / 1e6, colors[k])
          plt.subplot(nrow, ncol, 6)
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_t=ind_t)
          plt.plot(xt_, yt_, "o" + colors[k])
          plt.plot(lins_h, y[:, 1] / 1e-4, colors[k])
  
          ind_t = ind_t_2
          x = get_x(ind_M=ind_M, ind_t=ind_t)
          y = interp.predict_values(x)
  
          plt.subplot(nrow, ncol, 7)
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_t=ind_t)
          plt.plot(xt_, yt_, "o" + colors[k])
          plt.plot(lins_h, y[:, 0] / 1e6, colors[k])
          plt.subplot(nrow, ncol, 8)
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_t=ind_t)
          plt.plot(xt_, yt_, "o" + colors[k])
          plt.plot(lins_h, y[:, 1] / 1e-4, colors[k])
  
      # -----------------------------------------------------------------------------
  
      # Mach number slices
      for k, ind_h in enumerate(ind_h_list):
          ind_t = ind_t_1
          x = get_x(ind_t=ind_t, ind_h=ind_h)
          y = interp.predict_values(x)
  
          plt.subplot(nrow, ncol, 9)
          xt_, yt_ = get_pts(xt, yt, 0, ind_h=ind_h, ind_t=ind_t)
          plt.plot(xt_, yt_, "o" + colors[k])
          plt.plot(lins_M, y[:, 0] / 1e6, colors[k])
          plt.subplot(nrow, ncol, 10)
          xt_, yt_ = get_pts(xt, yt, 1, ind_h=ind_h, ind_t=ind_t)
          plt.plot(xt_, yt_, "o" + colors[k])
          plt.plot(lins_M, y[:, 1] / 1e-4, colors[k])
  
          ind_t = ind_t_2
          x = get_x(ind_t=ind_t, ind_h=ind_h)
          y = interp.predict_values(x)
  
          plt.subplot(nrow, ncol, 11)
          xt_, yt_ = get_pts(xt, yt, 0, ind_h=ind_h, ind_t=ind_t)
          plt.plot(xt_, yt_, "o" + colors[k])
          plt.plot(lins_M, y[:, 0] / 1e6, colors[k])
          plt.subplot(nrow, ncol, 12)
          xt_, yt_ = get_pts(xt, yt, 1, ind_h=ind_h, ind_t=ind_t)
          plt.plot(xt_, yt_, "o" + colors[k])
          plt.plot(lins_M, y[:, 1] / 1e-4, colors[k])
  
      # -----------------------------------------------------------------------------
  
      for k in range(4):
          legend_entries = []
          for ind_h in ind_h_list:
              legend_entries.append("h={}".format(val_h[ind_h]))
              legend_entries.append("")
  
          plt.subplot(nrow, ncol, k + 1)
          plt.legend(legend_entries)
  
          plt.subplot(nrow, ncol, k + 9)
          plt.legend(legend_entries)
  
          legend_entries = []
          for ind_M in ind_M_list:
              legend_entries.append("M={}".format(val_M[ind_M]))
              legend_entries.append("")
  
          plt.subplot(nrow, ncol, k + 5)
          plt.legend(legend_entries)
  
      plt.show()
  

RMTB
----

.. code-block:: python

  from smt.surrogate_models import RMTB
  from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine
  
  xt, yt, dyt_dxt, xlimits = get_b777_engine()
  
  interp = RMTB(
      num_ctrl_pts=15,
      xlimits=xlimits,
      nonlinear_maxiter=20,
      approx_order=2,
      energy_weight=0e-14,
      regularization_weight=0e-18,
      extrapolate=True,
  )
  interp.set_training_values(xt, yt)
  interp.set_training_derivatives(xt, dyt_dxt[:, :, 0], 0)
  interp.set_training_derivatives(xt, dyt_dxt[:, :, 1], 1)
  interp.set_training_derivatives(xt, dyt_dxt[:, :, 2], 2)
  interp.train()
  
  plot_b777_engine(xt, yt, xlimits, interp)
  
::

  ___________________________________________________________________________
     
                                     RMTB
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 1056
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Pre-computing matrices ...
           Computing dof2coeff ...
           Computing dof2coeff - done. Time (sec):  0.0000000
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.2652001
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0000000
        Pre-computing matrices - done. Time (sec):  0.2652001
        Solving for degrees of freedom ...
           Solving initial startup problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.857178281e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.364349733e+05 7.002441710e+09
              Solving for output 0 - done. Time (sec):  0.0936000
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.711896708e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.384257034e-03 3.512467641e-07
              Solving for output 1 - done. Time (sec):  0.0936000
           Solving initial startup problem (n=3375) - done. Time (sec):  0.1872001
           Solving nonlinear problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.364349733e+05 7.002441710e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.401682427e+04 1.956585489e+09
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.640761309e+04 5.653768085e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.726949662e+04 3.860194807e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.244331543e+04 3.735217325e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.356309977e+04 3.232040667e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.896770441e+04 2.970854602e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.168979712e+04 2.643923864e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.199133401e+04 2.223771115e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 9.363877631e+03 2.013234589e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 9.544160641e+03 1.861724031e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 9.458916793e+03 1.762819815e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 4.152198214e+03 1.661887141e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 8.359804107e+03 1.619868009e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 2.678073894e+03 1.599839425e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 2.301049932e+03 1.583627245e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 3.127472449e+03 1.554361115e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 2.879195835e+03 1.516054749e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 1.583184160e+03 1.493412967e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 2.202973513e+03 1.492035778e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.397841194e+03 1.489828558e+08
              Solving for output 0 - done. Time (sec):  1.7159998
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.384257034e-03 3.512467641e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.575138262e-04 6.166597300e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.156992731e-04 1.817140551e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.070220585e-04 8.504635606e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 1.711558893e-04 7.824284644e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 1.147466159e-04 6.729973912e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 1.033293877e-04 5.063463186e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 5.272698157e-05 2.929839938e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 4.894442104e-05 2.071717930e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 2.850823295e-05 1.797321609e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.566163204e-05 1.713105879e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.728118053e-05 1.606498899e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.407731298e-05 1.439553327e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.588414550e-05 1.302254672e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.941516089e-05 1.258276496e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.159190980e-05 1.239434907e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.872674427e-05 1.235569556e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 1.169536710e-05 1.206341167e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.005666171e-05 1.172498758e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 4.240888944e-06 1.143928197e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 4.653082813e-06 1.142989811e-09
              Solving for output 1 - done. Time (sec):  1.7472000
           Solving nonlinear problem (n=3375) - done. Time (sec):  3.4631999
        Solving for degrees of freedom - done. Time (sec):  3.6503999
     Training - done. Time (sec):  3.9156001
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0156000
     
     Prediction time/pt. (sec) :  0.0001560
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0156000
     
     Prediction time/pt. (sec) :  0.0001560
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: b777_engine.png
  :scale: 60 %
  :align: center

RMTC
----

.. code-block:: python

  from smt.surrogate_models import RMTC
  from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine
  
  xt, yt, dyt_dxt, xlimits = get_b777_engine()
  
  interp = RMTC(
      num_elements=6,
      xlimits=xlimits,
      nonlinear_maxiter=20,
      approx_order=2,
      energy_weight=0.0,
      regularization_weight=0.0,
      extrapolate=True,
  )
  interp.set_training_values(xt, yt)
  interp.set_training_derivatives(xt, dyt_dxt[:, :, 0], 0)
  interp.set_training_derivatives(xt, dyt_dxt[:, :, 1], 1)
  interp.set_training_derivatives(xt, dyt_dxt[:, :, 2], 2)
  interp.train()
  
  plot_b777_engine(xt, yt, xlimits, interp)
  
::

  ___________________________________________________________________________
     
                                     RMTC
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 1056
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Pre-computing matrices ...
           Computing dof2coeff ...
           Computing dof2coeff - done. Time (sec):  0.0311999
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.1716001
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0624001
        Pre-computing matrices - done. Time (sec):  0.2652001
        Solving for degrees of freedom ...
           Solving initial startup problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.864862172e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.301220784e+05 2.043089744e+09
              Solving for output 0 - done. Time (sec):  0.1872001
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 8.095040141e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.247766422e-03 1.322502818e-07
              Solving for output 1 - done. Time (sec):  0.2027998
           Solving initial startup problem (n=2744) - done. Time (sec):  0.3899999
           Solving nonlinear problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.301220784e+05 2.043089744e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.853768988e+04 4.204382514e+08
                 Iteration (num., iy, grad. norm, func.) :   1   0 1.600292441e+04 3.528182269e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 2.192959017e+04 3.499245939e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 8.908430676e+03 3.371333491e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 4.826696294e+03 3.326895469e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 4.466377088e+03 3.320607428e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.811936973e+03 3.312893629e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.939207818e+03 3.307236804e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.606864853e+03 3.304685748e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.876454015e+03 3.303459940e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 1.381228599e+03 3.302005814e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.427468675e+03 3.301258329e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 8.863567115e+02 3.300062354e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 8.708862351e+02 3.299010976e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 4.801718324e+02 3.298332669e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 4.188928791e+02 3.298207697e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 4.809479966e+02 3.298126567e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 7.635186662e+02 3.298073881e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 4.534280606e+02 3.298003186e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 3.973724388e+02 3.297955330e+08
              Solving for output 0 - done. Time (sec):  3.9624000
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.247766422e-03 1.322502818e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.967550240e-04 9.514176202e-09
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.022974213e-04 7.902718497e-09
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.940967866e-04 6.064696564e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 9.190272069e-05 4.306045763e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 9.362879272e-05 4.066039150e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 7.167971812e-05 3.747268947e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 4.524808243e-05 3.367699068e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 3.853416937e-05 3.209181099e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 4.232980316e-05 3.129247089e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 3.190371873e-05 3.067320241e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 1.974177570e-05 3.040505234e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.881369844e-05 3.034137061e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.436660531e-05 3.012771286e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.788606605e-05 2.992655580e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.152719843e-05 2.958698604e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.156807011e-05 2.937664628e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 8.045689579e-06 2.928032775e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.163004012e-05 2.926867367e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 8.598124448e-06 2.924036478e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 6.696045882e-06 2.923305126e-09
              Solving for output 1 - done. Time (sec):  3.9780002
           Solving nonlinear problem (n=2744) - done. Time (sec):  7.9404001
        Solving for degrees of freedom - done. Time (sec):  8.3304000
     Training - done. Time (sec):  8.5956001
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0156000
     
     Prediction time/pt. (sec) :  0.0001560
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0156000
     
     Prediction time/pt. (sec) :  0.0001560
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: b777_engine.png
  :scale: 60 %
  :align: center
