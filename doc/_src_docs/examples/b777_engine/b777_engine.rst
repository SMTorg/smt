Boeing 777 engine data set
==========================

.. code-block:: python

  import os
  
  import numpy as np
  
  
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
      import matplotlib
      import numpy as np
  
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
  
      fig, axs = plt.subplots(nrow, ncol, gridspec_kw={"hspace": 0.5}, figsize=(15, 25))
  
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

  from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine
  from smt.surrogate_models import RMTB
  
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
           Computing energy terms - done. Time (sec):  0.1866009
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0105088
        Pre-computing matrices - done. Time (sec):  0.1971097
        Solving for degrees of freedom ...
           Solving initial startup problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.857178281e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.373632370e+05 6.994943224e+09
              Solving for output 0 - done. Time (sec):  0.0729773
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.711896708e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.379731698e-03 3.522836449e-07
              Solving for output 1 - done. Time (sec):  0.0662329
           Solving initial startup problem (n=3375) - done. Time (sec):  0.1392102
           Solving nonlinear problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.373632370e+05 6.994943224e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.228762039e+04 1.953390951e+09
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.731024753e+04 5.658049284e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.751054702e+04 3.886958030e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.359943786e+04 3.770962330e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.551184108e+04 3.265923277e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.763819113e+04 3.006483951e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.806072295e+04 2.662704404e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 8.625485033e+03 2.230481305e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.042753219e+04 2.025104184e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 8.520289391e+03 1.872411238e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 8.723807262e+03 1.766506426e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 7.552188302e+03 1.659406243e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 5.805156511e+03 1.622801583e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 3.760772225e+03 1.607839480e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 5.190020572e+03 1.603716663e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 3.526170085e+03 1.569770171e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 3.601650746e+03 1.535065970e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 1.450796571e+03 1.500561151e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 2.465781060e+03 1.490289858e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.706970844e+03 1.487533340e+08
              Solving for output 0 - done. Time (sec):  1.3641312
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.379731698e-03 3.522836449e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.549321664e-04 6.189348135e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.033753847e-04 1.813290183e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.074294700e-04 8.416635233e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 1.679717689e-04 7.772256041e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 1.210384969e-04 6.709066740e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 1.013896403e-04 5.088687968e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 4.653467395e-05 2.986912993e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 5.102746264e-05 2.090009968e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 2.082183811e-05 1.785990436e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.819521568e-05 1.699600540e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 1.920704410e-05 1.598595555e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.286626885e-05 1.443383083e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.701209076e-05 1.308124029e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.559856977e-05 1.256780788e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 9.245693721e-06 1.231816967e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 9.360150345e-06 1.216232750e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 9.022769375e-06 1.188176708e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 7.918364377e-06 1.155510003e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 5.134927534e-06 1.140449521e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 6.357993355e-06 1.139634498e-09
              Solving for output 1 - done. Time (sec):  1.6146402
           Solving nonlinear problem (n=3375) - done. Time (sec):  2.9787714
        Solving for degrees of freedom - done. Time (sec):  3.1179817
     Training - done. Time (sec):  3.3150914
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
     Predicting - done. Time (sec):  0.0009975
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009973
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009985
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009966
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009973
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005682
     
     Prediction time/pt. (sec) :  0.0000057
     
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
     Predicting - done. Time (sec):  0.0084059
     
     Prediction time/pt. (sec) :  0.0000841
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0024045
     
     Prediction time/pt. (sec) :  0.0000240
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0024514
     
     Prediction time/pt. (sec) :  0.0000245
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009983
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009975
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009966
     
     Prediction time/pt. (sec) :  0.0000100
     
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

  from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine
  from smt.surrogate_models import RMTC
  
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
           Computing dof2coeff - done. Time (sec):  0.0305967
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.1891723
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0523190
        Pre-computing matrices - done. Time (sec):  0.2720881
        Solving for degrees of freedom ...
           Solving initial startup problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.864862172e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.029343879e+05 2.066646848e+09
              Solving for output 0 - done. Time (sec):  0.1414769
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 8.095040141e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.333265997e-03 1.320416078e-07
              Solving for output 1 - done. Time (sec):  0.1417410
           Solving initial startup problem (n=2744) - done. Time (sec):  0.2832179
           Solving nonlinear problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.029343879e+05 2.066646848e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 3.382217141e+04 4.205527351e+08
                 Iteration (num., iy, grad. norm, func.) :   1   0 1.715751811e+04 3.531406723e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 1.961408652e+04 3.503658706e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 1.071978888e+04 3.373032349e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 4.813663988e+03 3.327146065e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 5.535339851e+03 3.320808460e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 4.310257739e+03 3.312895022e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.970509617e+03 3.307174337e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 2.087065844e+03 3.304717263e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.991997699e+03 3.303446843e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 9.702487316e+02 3.301998830e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.477775817e+03 3.301248228e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 8.216315742e+02 3.300010023e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 1.345080167e+03 3.298973205e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 4.575430141e+02 3.298315569e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 3.763188410e+02 3.298202017e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 7.044825455e+02 3.298133312e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 5.175024952e+02 3.298089323e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 5.659469259e+02 3.298071432e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 3.408424643e+02 3.298015516e+08
              Solving for output 0 - done. Time (sec):  2.9187315
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.333265997e-03 1.320416078e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.969689807e-04 9.494233834e-09
                 Iteration (num., iy, grad. norm, func.) :   1   1 2.969731939e-04 7.887883745e-09
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.611035743e-04 6.071186215e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 9.825052294e-05 4.309177186e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 9.008178910e-05 4.064996698e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 7.008779861e-05 3.745524434e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 4.156123324e-05 3.366608292e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 3.961930390e-05 3.208685455e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 4.319438376e-05 3.128213736e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.746662124e-05 3.067522541e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.215321541e-05 3.041279415e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 4.748356730e-05 3.032369678e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.687771835e-05 3.010213288e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 2.626855139e-05 2.989854076e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.125863347e-05 2.956369574e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.252003690e-05 2.937041725e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 9.968771466e-06 2.929476869e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.487456551e-05 2.926019676e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 5.180815992e-06 2.920834147e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 3.876107086e-06 2.920143941e-09
              Solving for output 1 - done. Time (sec):  2.8581350
           Solving nonlinear problem (n=2744) - done. Time (sec):  5.7768664
        Solving for degrees of freedom - done. Time (sec):  6.0600843
     Training - done. Time (sec):  6.3406084
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0020473
     
     Prediction time/pt. (sec) :  0.0000205
     
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
     Predicting - done. Time (sec):  0.0080781
     
     Prediction time/pt. (sec) :  0.0000808
     
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
     Predicting - done. Time (sec):  0.0020542
     
     Prediction time/pt. (sec) :  0.0000205
     
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
     Predicting - done. Time (sec):  0.0080700
     
     Prediction time/pt. (sec) :  0.0000807
     
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
     Predicting - done. Time (sec):  0.0080240
     
     Prediction time/pt. (sec) :  0.0000802
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0020549
     
     Prediction time/pt. (sec) :  0.0000205
     
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
     Predicting - done. Time (sec):  0.0080700
     
     Prediction time/pt. (sec) :  0.0000807
     
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
     Predicting - done. Time (sec):  0.0020545
     
     Prediction time/pt. (sec) :  0.0000205
     
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
     Predicting - done. Time (sec):  0.0080791
     
     Prediction time/pt. (sec) :  0.0000808
     
  
.. figure:: b777_engine.png
  :scale: 60 %
  :align: center
