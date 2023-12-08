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
           Computing dof2coeff - done. Time (sec):  0.0000029
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0005333
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.2833576
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0102527
        Pre-computing matrices - done. Time (sec):  0.2942176
        Solving for degrees of freedom ...
           Solving initial startup problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.857178281e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.364349733e+05 7.002441710e+09
              Solving for output 0 - done. Time (sec):  0.0885880
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.711896708e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.384257034e-03 3.512467641e-07
              Solving for output 1 - done. Time (sec):  0.0884008
           Solving initial startup problem (n=3375) - done. Time (sec):  0.1770794
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
              Solving for output 0 - done. Time (sec):  1.7724838
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
              Solving for output 1 - done. Time (sec):  1.7775493
           Solving nonlinear problem (n=3375) - done. Time (sec):  3.5501003
        Solving for degrees of freedom - done. Time (sec):  3.7272718
     Training - done. Time (sec):  4.0223479
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012805
     
     Prediction time/pt. (sec) :  0.0000128
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0011656
     
     Prediction time/pt. (sec) :  0.0000117
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0011556
     
     Prediction time/pt. (sec) :  0.0000116
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0011547
     
     Prediction time/pt. (sec) :  0.0000115
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0011721
     
     Prediction time/pt. (sec) :  0.0000117
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0011621
     
     Prediction time/pt. (sec) :  0.0000116
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0011656
     
     Prediction time/pt. (sec) :  0.0000117
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0011466
     
     Prediction time/pt. (sec) :  0.0000115
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012228
     
     Prediction time/pt. (sec) :  0.0000122
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012078
     
     Prediction time/pt. (sec) :  0.0000121
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012228
     
     Prediction time/pt. (sec) :  0.0000122
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012078
     
     Prediction time/pt. (sec) :  0.0000121
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012197
     
     Prediction time/pt. (sec) :  0.0000122
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012212
     
     Prediction time/pt. (sec) :  0.0000122
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012045
     
     Prediction time/pt. (sec) :  0.0000120
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012193
     
     Prediction time/pt. (sec) :  0.0000122
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012119
     
     Prediction time/pt. (sec) :  0.0000121
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012147
     
     Prediction time/pt. (sec) :  0.0000121
     
  
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
           Computing dof2coeff - done. Time (sec):  0.0221884
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0005217
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.1968417
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0910423
        Pre-computing matrices - done. Time (sec):  0.3106997
        Solving for degrees of freedom ...
           Solving initial startup problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.864862172e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.020804204e+05 2.067017787e+09
              Solving for output 0 - done. Time (sec):  0.1912801
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 8.095040141e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.242052177e-03 1.322622537e-07
              Solving for output 1 - done. Time (sec):  0.1920962
           Solving initial startup problem (n=2744) - done. Time (sec):  0.3834834
           Solving nonlinear problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.020804204e+05 2.067017787e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 3.795378678e+04 4.209003076e+08
                 Iteration (num., iy, grad. norm, func.) :   1   0 1.691388107e+04 3.530609622e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 2.424674299e+04 3.502442863e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 1.058301522e+04 3.371492582e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 4.312724954e+03 3.326822270e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 5.977501584e+03 3.320622264e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.564093348e+03 3.312761931e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.633911637e+03 3.307130145e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.496166349e+03 3.304586905e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.842768951e+03 3.303544335e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 1.051264756e+03 3.302209836e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.324634199e+03 3.301346076e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 9.434603444e+02 3.299980929e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 8.313796402e+02 3.299030047e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 5.486546277e+02 3.298453756e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 6.551215009e+02 3.298396348e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 4.500885038e+02 3.298378066e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 1.028007688e+03 3.298282370e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 3.399838938e+02 3.298087400e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 2.053750868e+02 3.298071682e+08
              Solving for output 0 - done. Time (sec):  3.7802360
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.242052177e-03 1.322622537e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.336667748e-04 9.461988364e-09
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.413078795e-04 7.888237722e-09
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.136195000e-04 6.077625043e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 8.542812525e-05 4.320340731e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 8.048002545e-05 4.069045408e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 5.885314829e-05 3.747906307e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 4.591463641e-05 3.368137289e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 3.900587215e-05 3.208417254e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 4.559233031e-05 3.125987804e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.617598725e-05 3.067238678e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 3.747446115e-05 3.046046092e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.361552341e-05 3.036950106e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 2.753651777e-05 3.019194609e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 2.186673375e-05 2.992432238e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.772397035e-05 2.960899015e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.070987545e-05 2.936477461e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 9.065950919e-06 2.926874777e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 7.400482695e-06 2.924517350e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 6.317494830e-06 2.922681890e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 1.047966941e-05 2.918951548e-09
              Solving for output 1 - done. Time (sec):  3.7826016
           Solving nonlinear problem (n=2744) - done. Time (sec):  7.5629272
        Solving for degrees of freedom - done. Time (sec):  7.9465067
     Training - done. Time (sec):  8.2599399
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0029576
     
     Prediction time/pt. (sec) :  0.0000296
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0028090
     
     Prediction time/pt. (sec) :  0.0000281
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0028059
     
     Prediction time/pt. (sec) :  0.0000281
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027966
     
     Prediction time/pt. (sec) :  0.0000280
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0029519
     
     Prediction time/pt. (sec) :  0.0000295
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0029473
     
     Prediction time/pt. (sec) :  0.0000295
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027869
     
     Prediction time/pt. (sec) :  0.0000279
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0025973
     
     Prediction time/pt. (sec) :  0.0000260
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0028708
     
     Prediction time/pt. (sec) :  0.0000287
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027230
     
     Prediction time/pt. (sec) :  0.0000272
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027533
     
     Prediction time/pt. (sec) :  0.0000275
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0026200
     
     Prediction time/pt. (sec) :  0.0000262
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0028691
     
     Prediction time/pt. (sec) :  0.0000287
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027015
     
     Prediction time/pt. (sec) :  0.0000270
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0028729
     
     Prediction time/pt. (sec) :  0.0000287
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027158
     
     Prediction time/pt. (sec) :  0.0000272
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0030129
     
     Prediction time/pt. (sec) :  0.0000301
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0028460
     
     Prediction time/pt. (sec) :  0.0000285
     
  
.. figure:: b777_engine.png
  :scale: 60 %
  :align: center
