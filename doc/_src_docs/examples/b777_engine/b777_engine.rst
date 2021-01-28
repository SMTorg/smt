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
           Computing energy terms - done. Time (sec):  0.2400000
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0099998
        Pre-computing matrices - done. Time (sec):  0.2499998
        Solving for degrees of freedom ...
           Solving initial startup problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.857178281e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.371838165e+05 6.993448074e+09
              Solving for output 0 - done. Time (sec):  0.0800002
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.711896708e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.374254361e-03 3.512412267e-07
              Solving for output 1 - done. Time (sec):  0.0900002
           Solving initial startup problem (n=3375) - done. Time (sec):  0.1700003
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
              Solving for output 0 - done. Time (sec):  1.6399999
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
              Solving for output 1 - done. Time (sec):  1.6299999
           Solving nonlinear problem (n=3375) - done. Time (sec):  3.2699997
        Solving for degrees of freedom - done. Time (sec):  3.4400001
     Training - done. Time (sec):  3.6899998
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
           Computing approximation terms - done. Time (sec):  0.0700002
        Pre-computing matrices - done. Time (sec):  0.2600000
        Solving for degrees of freedom ...
           Solving initial startup problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.864862172e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.273624644e+05 2.051869472e+09
              Solving for output 0 - done. Time (sec):  0.1800001
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 8.095040141e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.274404093e-03 1.321915975e-07
              Solving for output 1 - done. Time (sec):  0.1799998
           Solving initial startup problem (n=2744) - done. Time (sec):  0.3599999
           Solving nonlinear problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.273624644e+05 2.051869472e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 3.377295551e+04 4.199084241e+08
                 Iteration (num., iy, grad. norm, func.) :   1   0 2.092099507e+04 3.528490929e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 2.086992896e+04 3.498730463e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 8.548097295e+03 3.371447225e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 5.237579230e+03 3.326949260e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 3.958390673e+03 3.320806133e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.507130982e+03 3.313167741e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.993035210e+03 3.307306695e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.676891519e+03 3.304720174e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 2.024420665e+03 3.303435674e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 1.322134240e+03 3.301958320e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.525903295e+03 3.301195417e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 8.119954513e+02 3.299960494e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 1.309010125e+03 3.299057981e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 4.043930047e+02 3.298477029e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 5.438859352e+02 3.298397904e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 5.180648493e+02 3.298340628e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 7.926240522e+02 3.298326853e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 3.414678241e+02 3.298226373e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 2.098633328e+02 3.298173835e+08
              Solving for output 0 - done. Time (sec):  3.6400001
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.274404093e-03 1.321915975e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.693761004e-04 9.488550947e-09
                 Iteration (num., iy, grad. norm, func.) :   1   1 2.993728844e-04 7.911979754e-09
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.477171383e-04 6.091814506e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 1.126174787e-04 4.311968257e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 8.308096517e-05 4.064796121e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 6.168675408e-05 3.747988363e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 6.310008023e-05 3.360863223e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 3.326822995e-05 3.198494129e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 3.926180736e-05 3.122183257e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.052145934e-05 3.064119332e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.059432618e-05 3.043330234e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.653650832e-05 3.035548483e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 2.250101952e-05 3.013663166e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 2.209536584e-05 2.989061921e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.312821520e-05 2.953364268e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 9.726447224e-06 2.935407678e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 1.106722997e-05 2.928938589e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.121406275e-05 2.927980590e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 7.886316816e-06 2.924781972e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 6.485367416e-06 2.923750498e-09
              Solving for output 1 - done. Time (sec):  3.6400001
           Solving nonlinear problem (n=2744) - done. Time (sec):  7.2800002
        Solving for degrees of freedom - done. Time (sec):  7.6400001
     Training - done. Time (sec):  7.9099998
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
