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
           Computing dof2coeff - done. Time (sec):  0.0000000
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.2800000
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0100000
        Pre-computing matrices - done. Time (sec):  0.2900000
        Solving for degrees of freedom ...
           Solving initial startup problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.857178281e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.371838165e+05 6.993448074e+09
              Solving for output 0 - done. Time (sec):  0.0899999
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.711896708e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.374254361e-03 3.512412267e-07
              Solving for output 1 - done. Time (sec):  0.0799999
           Solving initial startup problem (n=3375) - done. Time (sec):  0.1699998
           Solving nonlinear problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.371838165e+05 6.993448074e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.296273998e+04 1.952753642e+09
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.699413853e+04 5.656526603e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.630530299e+04 3.867957232e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.132112066e+04 3.751114847e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.622497401e+04 3.233995367e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.632874586e+04 2.970251901e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.945828556e+04 2.644866975e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 7.031298294e+03 2.202296484e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 9.584176762e+03 2.010315328e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 6.066765838e+03 1.861519410e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 8.660248230e+03 1.774578981e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 5.959258886e+03 1.676188728e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 5.420215086e+03 1.611620261e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 3.047173881e+03 1.577535102e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 2.799164640e+03 1.565435585e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 3.613557675e+03 1.539615598e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 2.225395170e+03 1.512405059e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 2.043434052e+03 1.495341032e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 1.794946303e+03 1.489703531e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.362610245e+03 1.486885609e+08
              Solving for output 0 - done. Time (sec):  1.7100003
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.374254361e-03 3.512412267e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.525988536e-04 6.188393994e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.057617946e-04 1.809764195e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 1.607604906e-04 8.481761160e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 1.400165707e-04 7.796139756e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 1.120967416e-04 6.716864259e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 1.164592243e-04 5.027554636e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 3.837990633e-05 2.869982182e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 5.592710013e-05 2.076645667e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 2.315187099e-05 1.822951599e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.598674619e-05 1.718454630e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.476820921e-05 1.578553884e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.173672750e-05 1.417305155e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.113086815e-05 1.297183916e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.971817939e-05 1.259939268e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.181193289e-05 1.237157107e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.205550947e-05 1.234213169e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 1.274996177e-05 1.207112873e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.006884757e-05 1.173920482e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 4.804334068e-06 1.146309226e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 4.607661308e-06 1.143931945e-09
              Solving for output 1 - done. Time (sec):  1.7609999
           Solving nonlinear problem (n=3375) - done. Time (sec):  3.4710002
        Solving for degrees of freedom - done. Time (sec):  3.6410000
     Training - done. Time (sec):  3.9310000
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
     Predicting - done. Time (sec):  0.0099998
     
     Prediction time/pt. (sec) :  0.0001000
     
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
     Predicting - done. Time (sec):  0.0100002
     
     Prediction time/pt. (sec) :  0.0001000
     
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
     Predicting - done. Time (sec):  0.0099998
     
     Prediction time/pt. (sec) :  0.0001000
     
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
           Computing dof2coeff - done. Time (sec):  0.0200000
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.1699998
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0599999
        Pre-computing matrices - done. Time (sec):  0.2499998
        Solving for degrees of freedom ...
           Solving initial startup problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.864862172e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.954376733e+05 2.069307906e+09
              Solving for output 0 - done. Time (sec):  0.2000000
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 8.095040141e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.232503686e-03 1.322818515e-07
              Solving for output 1 - done. Time (sec):  0.1800001
           Solving initial startup problem (n=2744) - done. Time (sec):  0.3800001
           Solving nonlinear problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.954376733e+05 2.069307906e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 3.688989036e+04 4.210539534e+08
                 Iteration (num., iy, grad. norm, func.) :   1   0 1.711008392e+04 3.530483452e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 1.922847114e+04 3.504226099e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 9.537041405e+03 3.373561783e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 4.760722218e+03 3.327296410e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 6.376817973e+03 3.320587914e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.785458727e+03 3.312772871e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.159194888e+03 3.307195890e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.756134897e+03 3.304676354e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 2.544893553e+03 3.303514392e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 9.687040825e+02 3.302102085e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.529434541e+03 3.301310381e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 8.575442103e+02 3.300046689e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 7.946674618e+02 3.299019198e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 4.138160638e+02 3.298361385e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 3.344866005e+02 3.298229530e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 4.791739707e+02 3.298135435e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 7.683854023e+02 3.298069892e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 4.087822635e+02 3.298060925e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 2.867885276e+02 3.298024065e+08
              Solving for output 0 - done. Time (sec):  3.8799999
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.232503686e-03 1.322818515e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 4.036053441e-04 9.595113849e-09
                 Iteration (num., iy, grad. norm, func.) :   1   1 2.706482615e-04 7.876088023e-09
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.260119896e-04 6.102248425e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 1.174679565e-04 4.321455879e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 1.073558503e-04 4.060995795e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 6.275682902e-05 3.739333496e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 5.607076749e-05 3.359307617e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 3.901842383e-05 3.200980932e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 4.898053749e-05 3.121087196e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.789456822e-05 3.062451544e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.227200280e-05 3.044681794e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 3.096887425e-05 3.026565623e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.606775960e-05 2.993510820e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.554076288e-05 2.978354868e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.041250866e-05 2.953574603e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.155923771e-05 2.935360717e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 9.466047104e-06 2.927068596e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 9.185257136e-06 2.924961044e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 7.396047208e-06 2.924123759e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 1.605557201e-05 2.921915424e-09
              Solving for output 1 - done. Time (sec):  3.8500001
           Solving nonlinear problem (n=2744) - done. Time (sec):  7.7300000
        Solving for degrees of freedom - done. Time (sec):  8.1100001
     Training - done. Time (sec):  8.3700001
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
     Predicting - done. Time (sec):  0.0100000
     
     Prediction time/pt. (sec) :  0.0001000
     
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
     Predicting - done. Time (sec):  0.0100000
     
     Prediction time/pt. (sec) :  0.0001000
     
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
     Predicting - done. Time (sec):  0.0100002
     
     Prediction time/pt. (sec) :  0.0001000
     
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
     Predicting - done. Time (sec):  0.0099998
     
     Prediction time/pt. (sec) :  0.0001000
     
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
     Predicting - done. Time (sec):  0.0099998
     
     Prediction time/pt. (sec) :  0.0001000
     
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
     Predicting - done. Time (sec):  0.0100002
     
     Prediction time/pt. (sec) :  0.0001000
     
  
.. figure:: b777_engine.png
  :scale: 60 %
  :align: center
