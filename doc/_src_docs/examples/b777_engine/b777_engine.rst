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
  
      # --------------------
  
      fig, axs = plt.subplots(6, 2, gridspec_kw={"hspace": 0.5}, figsize=(15, 25))
  
      axs[0, 0].set_title("M={}".format(val_M[ind_M_1]))
      axs[0, 0].set(xlabel="throttle", ylabel="thrust (x 1e6 N)")
  
      axs[0, 1].set_title("M={}".format(val_M[ind_M_1]))
      axs[0, 1].set(xlabel="throttle", ylabel="SFC (x 1e-3 N/N/s)")
  
      axs[1, 0].set_title("M={}".format(val_M[ind_M_2]))
      axs[1, 0].set(xlabel="throttle", ylabel="thrust (x 1e6 N)")
  
      axs[1, 1].set_title("M={}".format(val_M[ind_M_2]))
      axs[1, 1].set(xlabel="throttle", ylabel="SFC (x 1e-3 N/N/s)")
  
      # --------------------
  
      axs[2, 0].set_title("throttle={}".format(val_t[ind_t_1]))
      axs[2, 0].set(xlabel="altitude (km)", ylabel="thrust (x 1e6 N)")
  
      axs[2, 1].set_title("throttle={}".format(val_t[ind_t_1]))
      axs[2, 1].set(xlabel="altitude (km)", ylabel="SFC (x 1e-3 N/N/s)")
  
      axs[3, 0].set_title("throttle={}".format(val_t[ind_t_2]))
      axs[3, 0].set(xlabel="altitude (km)", ylabel="thrust (x 1e6 N)")
  
      axs[3, 1].set_title("throttle={}".format(val_t[ind_t_2]))
      axs[3, 1].set(xlabel="altitude (km)", ylabel="SFC (x 1e-3 N/N/s)")
  
      # --------------------
  
      axs[4, 0].set_title("throttle={}".format(val_t[ind_t_1]))
      axs[4, 0].set(xlabel="Mach number", ylabel="thrust (x 1e6 N)")
  
      axs[4, 1].set_title("throttle={}".format(val_t[ind_t_1]))
      axs[4, 1].set(xlabel="Mach number", ylabel="SFC (x 1e-3 N/N/s)")
  
      axs[5, 0].set_title("throttle={}".format(val_t[ind_t_2]))
      axs[5, 0].set(xlabel="Mach number", ylabel="thrust (x 1e6 N)")
  
      axs[5, 1].set_title("throttle={}".format(val_t[ind_t_2]))
      axs[5, 1].set(xlabel="Mach number", ylabel="SFC (x 1e-3 N/N/s)")
  
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
  
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_h=ind_h)
          axs[0, 0].plot(xt_, yt_, "o" + colors[k])
          axs[0, 0].plot(lins_t, y[:, 0] / 1e6, colors[k])
  
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_h=ind_h)
          axs[0, 1].plot(xt_, yt_, "o" + colors[k])
          axs[0, 1].plot(lins_t, y[:, 1] / 1e-4, colors[k])
  
          ind_M = ind_M_2
          x = get_x(ind_M=ind_M, ind_h=ind_h)
          y = interp.predict_values(x)
  
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_h=ind_h)
          axs[1, 0].plot(xt_, yt_, "o" + colors[k])
          axs[1, 0].plot(lins_t, y[:, 0] / 1e6, colors[k])
  
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_h=ind_h)
          axs[1, 1].plot(xt_, yt_, "o" + colors[k])
          axs[1, 1].plot(lins_t, y[:, 1] / 1e-4, colors[k])
  
      # -----------------------------------------------------------------------------
  
      # Altitude slices
      for k, ind_M in enumerate(ind_M_list):
          ind_t = ind_t_1
          x = get_x(ind_M=ind_M, ind_t=ind_t)
          y = interp.predict_values(x)
  
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_t=ind_t)
          axs[2, 0].plot(xt_, yt_, "o" + colors[k])
          axs[2, 0].plot(lins_h, y[:, 0] / 1e6, colors[k])
  
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_t=ind_t)
          axs[2, 1].plot(xt_, yt_, "o" + colors[k])
          axs[2, 1].plot(lins_h, y[:, 1] / 1e-4, colors[k])
  
          ind_t = ind_t_2
          x = get_x(ind_M=ind_M, ind_t=ind_t)
          y = interp.predict_values(x)
  
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_t=ind_t)
          axs[3, 0].plot(xt_, yt_, "o" + colors[k])
          axs[3, 0].plot(lins_h, y[:, 0] / 1e6, colors[k])
  
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_t=ind_t)
          axs[3, 1].plot(xt_, yt_, "o" + colors[k])
          axs[3, 1].plot(lins_h, y[:, 1] / 1e-4, colors[k])
  
      # -----------------------------------------------------------------------------
  
      # Mach number slices
      for k, ind_h in enumerate(ind_h_list):
          ind_t = ind_t_1
          x = get_x(ind_t=ind_t, ind_h=ind_h)
          y = interp.predict_values(x)
  
          xt_, yt_ = get_pts(xt, yt, 0, ind_h=ind_h, ind_t=ind_t)
          axs[4, 0].plot(xt_, yt_, "o" + colors[k])
          axs[4, 0].plot(lins_M, y[:, 0] / 1e6, colors[k])
  
          xt_, yt_ = get_pts(xt, yt, 1, ind_h=ind_h, ind_t=ind_t)
          axs[4, 1].plot(xt_, yt_, "o" + colors[k])
          axs[4, 1].plot(lins_M, y[:, 1] / 1e-4, colors[k])
  
          ind_t = ind_t_2
          x = get_x(ind_t=ind_t, ind_h=ind_h)
          y = interp.predict_values(x)
  
          xt_, yt_ = get_pts(xt, yt, 0, ind_h=ind_h, ind_t=ind_t)
          axs[5, 0].plot(xt_, yt_, "o" + colors[k])
          axs[5, 0].plot(lins_M, y[:, 0] / 1e6, colors[k])
  
          xt_, yt_ = get_pts(xt, yt, 1, ind_h=ind_h, ind_t=ind_t)
          axs[5, 1].plot(xt_, yt_, "o" + colors[k])
          axs[5, 1].plot(lins_M, y[:, 1] / 1e-4, colors[k])
  
      # -----------------------------------------------------------------------------
  
      for k in range(2):
          legend_entries = []
          for ind_h in ind_h_list:
              legend_entries.append("h={}".format(val_h[ind_h]))
              legend_entries.append("")
  
          axs[k, 0].legend(legend_entries)
          axs[k, 1].legend(legend_entries)
  
          axs[k + 4, 0].legend(legend_entries)
          axs[k + 4, 1].legend(legend_entries)
  
          legend_entries = []
          for ind_M in ind_M_list:
              legend_entries.append("M={}".format(val_M[ind_M]))
              legend_entries.append("")
  
          axs[k + 2, 0].legend(legend_entries)
          axs[k + 2, 1].legend(legend_entries)
  
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
           Computing dof2coeff - done. Time (sec):  0.0000010
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0001478
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0946949
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0031919
        Pre-computing matrices - done. Time (sec):  0.0980628
        Solving for degrees of freedom ...
           Solving initial startup problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.857178281e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.293420813e+05 7.013297605e+09
              Solving for output 0 - done. Time (sec):  0.0417089
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.711896708e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.591831008e-03 3.481810730e-07
              Solving for output 1 - done. Time (sec):  0.0414689
           Solving initial startup problem (n=3375) - done. Time (sec):  0.0832350
           Solving nonlinear problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.293420813e+05 7.013297605e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.918153307e+04 1.946328603e+09
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.760511919e+04 5.642222750e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.509905718e+04 3.916910153e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.266254986e+04 3.812681287e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.433474999e+04 3.307625875e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.836831901e+04 3.044379206e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.369130884e+04 2.694361513e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.189289106e+04 2.253292276e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.348951232e+04 2.030170277e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 9.749280003e+03 1.871694192e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 7.933559876e+03 1.773445232e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 6.264482674e+03 1.676226215e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 8.774953813e+03 1.628378170e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 3.157944089e+03 1.602058800e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 2.774402118e+03 1.586039661e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 4.092421745e+03 1.557881701e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 4.055495308e+03 1.520351077e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 1.671677051e+03 1.497665190e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 2.100834104e+03 1.492367468e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.848493546e+03 1.492015420e+08
              Solving for output 0 - done. Time (sec):  0.8298440
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.591831008e-03 3.481810730e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.662651803e-04 6.180892232e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.003014390e-04 1.793235061e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 1.990534749e-04 8.250975285e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 1.623714955e-04 7.620571122e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 1.143352248e-04 6.621461165e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 9.551202845e-05 5.069094900e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 4.609982291e-05 2.979488927e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 4.024524802e-05 2.089470591e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 2.160993922e-05 1.783815797e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.575210915e-05 1.704417034e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.139027152e-05 1.618671689e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.613440336e-05 1.455574136e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.182699498e-05 1.311973808e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.342778429e-05 1.264965460e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 9.972369963e-06 1.245134658e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.799078033e-05 1.237777758e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 9.400245015e-06 1.203431806e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.153928975e-05 1.173616633e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 4.709245007e-06 1.148425945e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 7.972659809e-06 1.144617612e-09
              Solving for output 1 - done. Time (sec):  0.8405402
           Solving nonlinear problem (n=3375) - done. Time (sec):  1.6708009
        Solving for degrees of freedom - done. Time (sec):  1.7540612
     Training - done. Time (sec):  1.8526030
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0004449
     
     Prediction time/pt. (sec) :  0.0000044
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003459
     
     Prediction time/pt. (sec) :  0.0000035
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003281
     
     Prediction time/pt. (sec) :  0.0000033
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003288
     
     Prediction time/pt. (sec) :  0.0000033
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003409
     
     Prediction time/pt. (sec) :  0.0000034
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003412
     
     Prediction time/pt. (sec) :  0.0000034
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003278
     
     Prediction time/pt. (sec) :  0.0000033
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003269
     
     Prediction time/pt. (sec) :  0.0000033
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003259
     
     Prediction time/pt. (sec) :  0.0000033
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003278
     
     Prediction time/pt. (sec) :  0.0000033
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003219
     
     Prediction time/pt. (sec) :  0.0000032
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003238
     
     Prediction time/pt. (sec) :  0.0000032
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003328
     
     Prediction time/pt. (sec) :  0.0000033
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003290
     
     Prediction time/pt. (sec) :  0.0000033
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003281
     
     Prediction time/pt. (sec) :  0.0000033
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003271
     
     Prediction time/pt. (sec) :  0.0000033
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003283
     
     Prediction time/pt. (sec) :  0.0000033
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003231
     
     Prediction time/pt. (sec) :  0.0000032
     
  
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
           Computing dof2coeff - done. Time (sec):  0.0098188
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0001609
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0839014
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0360830
        Pre-computing matrices - done. Time (sec):  0.1300149
        Solving for degrees of freedom ...
           Solving initial startup problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.864862172e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.052317109e+05 2.069396782e+09
              Solving for output 0 - done. Time (sec):  0.1125259
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 8.095040141e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.273662871e-03 1.329452358e-07
              Solving for output 1 - done. Time (sec):  0.1014726
           Solving initial startup problem (n=2744) - done. Time (sec):  0.2145877
           Solving nonlinear problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.052317109e+05 2.069396782e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.996366083e+04 4.219824562e+08
                 Iteration (num., iy, grad. norm, func.) :   1   0 2.262504236e+04 3.529144002e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 2.631491561e+04 3.502952705e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 1.025035947e+04 3.373979635e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 5.035075808e+03 3.327895836e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 5.472210008e+03 3.320446921e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.744180795e+03 3.312471530e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.263098458e+03 3.307156644e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.386933927e+03 3.304724667e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 2.385520118e+03 3.303609813e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 1.218967550e+03 3.302206905e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.564832719e+03 3.301333245e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 7.375264475e+02 3.300031216e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 7.105096838e+02 3.299092956e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 5.063300804e+02 3.298500949e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 6.753465494e+02 3.298430233e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 5.894933008e+02 3.298417638e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 8.077216254e+02 3.298310170e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 3.516219330e+02 3.298104228e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 2.812559106e+02 3.298070074e+08
              Solving for output 0 - done. Time (sec):  1.8033929
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.273662871e-03 1.329452358e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 4.370977543e-04 9.590212540e-09
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.535273152e-04 8.038446158e-09
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.820354261e-04 6.088243046e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 1.018171021e-04 4.300815041e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 8.071151343e-05 4.077142295e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 7.162243229e-05 3.761864806e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 5.255169110e-05 3.370801596e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 4.368472782e-05 3.207626194e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 3.611768378e-05 3.127888941e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 4.734334534e-05 3.070805709e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.171968345e-05 3.040709354e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.696172674e-05 3.033491503e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 2.348198700e-05 3.011700927e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.947889207e-05 2.993904593e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.673014568e-05 2.964466175e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.446505038e-05 2.938848122e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 8.414603397e-06 2.925003780e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 6.862887355e-06 2.923601781e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 7.023657912e-06 2.922484797e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 9.554151439e-06 2.920588525e-09
              Solving for output 1 - done. Time (sec):  1.7982211
           Solving nonlinear problem (n=2744) - done. Time (sec):  3.6017787
        Solving for degrees of freedom - done. Time (sec):  3.8164029
     Training - done. Time (sec):  3.9474869
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0008461
     
     Prediction time/pt. (sec) :  0.0000085
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007920
     
     Prediction time/pt. (sec) :  0.0000079
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007811
     
     Prediction time/pt. (sec) :  0.0000078
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007799
     
     Prediction time/pt. (sec) :  0.0000078
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007579
     
     Prediction time/pt. (sec) :  0.0000076
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007420
     
     Prediction time/pt. (sec) :  0.0000074
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006809
     
     Prediction time/pt. (sec) :  0.0000068
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006998
     
     Prediction time/pt. (sec) :  0.0000070
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007617
     
     Prediction time/pt. (sec) :  0.0000076
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007932
     
     Prediction time/pt. (sec) :  0.0000079
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006812
     
     Prediction time/pt. (sec) :  0.0000068
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006924
     
     Prediction time/pt. (sec) :  0.0000069
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007968
     
     Prediction time/pt. (sec) :  0.0000080
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007973
     
     Prediction time/pt. (sec) :  0.0000080
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007963
     
     Prediction time/pt. (sec) :  0.0000080
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007989
     
     Prediction time/pt. (sec) :  0.0000080
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007648
     
     Prediction time/pt. (sec) :  0.0000076
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007770
     
     Prediction time/pt. (sec) :  0.0000078
     
  
.. figure:: b777_engine.png
  :scale: 60 %
  :align: center
