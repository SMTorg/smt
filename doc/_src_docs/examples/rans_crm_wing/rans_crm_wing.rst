RANS CRM wing 2-D data set
==========================

.. code-block:: python

  import numpy as np
  
  
  raw = np.array([
      [2.000000000000000000e+00 ,  4.500000000000000111e-01 ,  1.536799999999999972e-02 ,  3.674239999999999728e-01 ,  5.592279999999999474e-01 , -1.258039999999999992e-01 , -1.248699999999999984e-02],
      [3.500000000000000000e+00 ,  4.500000000000000111e-01 ,  1.985100000000000059e-02 ,  4.904470000000000218e-01 ,  7.574600000000000222e-01 , -1.615260000000000029e-01 ,  8.987000000000000197e-03],
      [5.000000000000000000e+00 ,  4.500000000000000111e-01 ,  2.571000000000000021e-02 ,  6.109189999999999898e-01 ,  9.497949999999999449e-01 , -1.954619999999999969e-01 ,  4.090900000000000092e-02],
      [6.500000000000000000e+00 ,  4.500000000000000111e-01 ,  3.304200000000000192e-02 ,  7.266120000000000356e-01 ,  1.131138999999999895e+00 , -2.255890000000000117e-01 ,  8.185399999999999621e-02],
      [8.000000000000000000e+00 ,  4.500000000000000111e-01 ,  4.318999999999999923e-02 ,  8.247250000000000414e-01 ,  1.271487000000000034e+00 , -2.397040000000000004e-01 ,  1.217659999999999992e-01],
      [0.000000000000000000e+00 ,  5.799999999999999600e-01 ,  1.136200000000000057e-02 ,  2.048760000000000026e-01 ,  2.950280000000000125e-01 , -7.882100000000000217e-02 , -2.280099999999999835e-02],
      [1.500000000000000000e+00 ,  5.799999999999999600e-01 ,  1.426000000000000011e-02 ,  3.375619999999999732e-01 ,  5.114130000000000065e-01 , -1.189420000000000061e-01 , -1.588200000000000028e-02],
      [3.000000000000000000e+00 ,  5.799999999999999600e-01 ,  1.866400000000000003e-02 ,  4.687450000000000228e-01 ,  7.240400000000000169e-01 , -1.577669999999999906e-01 ,  3.099999999999999891e-03],
      [4.500000000000000000e+00 ,  5.799999999999999600e-01 ,  2.461999999999999952e-02 ,  5.976639999999999731e-01 ,  9.311709999999999710e-01 , -1.944160000000000055e-01 ,  3.357500000000000068e-02],
      [6.000000000000000000e+00 ,  5.799999999999999600e-01 ,  3.280700000000000283e-02 ,  7.142249999999999988e-01 ,  1.111707999999999918e+00 , -2.205870000000000053e-01 ,  7.151699999999999724e-02],
      [0.000000000000000000e+00 ,  6.800000000000000488e-01 ,  1.138800000000000055e-02 ,  2.099310000000000065e-01 ,  3.032230000000000203e-01 , -8.187899999999999345e-02 , -2.172699999999999979e-02],
      [1.500000000000000000e+00 ,  6.800000000000000488e-01 ,  1.458699999999999927e-02 ,  3.518569999999999753e-01 ,  5.356630000000000003e-01 , -1.257649999999999879e-01 , -1.444800000000000077e-02],
      [3.000000000000000000e+00 ,  6.800000000000000488e-01 ,  1.952800000000000022e-02 ,  4.924879999999999813e-01 ,  7.644769999999999621e-01 , -1.678040000000000087e-01 ,  6.023999999999999841e-03],
      [4.500000000000000000e+00 ,  6.800000000000000488e-01 ,  2.666699999999999973e-02 ,  6.270339999999999803e-01 ,  9.801630000000000065e-01 , -2.035240000000000105e-01 ,  3.810000000000000192e-02],
      [6.000000000000000000e+00 ,  6.800000000000000488e-01 ,  3.891800000000000120e-02 ,  7.172730000000000494e-01 ,  1.097855999999999943e+00 , -2.014620000000000022e-01 ,  6.640000000000000069e-02],
      [0.000000000000000000e+00 ,  7.500000000000000000e-01 ,  1.150699999999999987e-02 ,  2.149069999999999869e-01 ,  3.115740000000000176e-01 , -8.498999999999999611e-02 , -2.057700000000000154e-02],
      [1.250000000000000000e+00 ,  7.500000000000000000e-01 ,  1.432600000000000019e-02 ,  3.415969999999999840e-01 ,  5.199390000000000400e-01 , -1.251009999999999900e-01 , -1.515400000000000080e-02],
      [2.500000000000000000e+00 ,  7.500000000000000000e-01 ,  1.856000000000000011e-02 ,  4.677589999999999804e-01 ,  7.262499999999999512e-01 , -1.635169999999999957e-01 ,  3.989999999999999949e-04],
      [3.750000000000000000e+00 ,  7.500000000000000000e-01 ,  2.472399999999999945e-02 ,  5.911459999999999493e-01 ,  9.254930000000000101e-01 , -1.966150000000000120e-01 ,  2.524900000000000061e-02],
      [5.000000000000000000e+00 ,  7.500000000000000000e-01 ,  3.506800000000000195e-02 ,  7.047809999999999908e-01 ,  1.097736000000000045e+00 , -2.143069999999999975e-01 ,  5.321300000000000335e-02],
      [0.000000000000000000e+00 ,  8.000000000000000444e-01 ,  1.168499999999999921e-02 ,  2.196390000000000009e-01 ,  3.197160000000000002e-01 , -8.798200000000000465e-02 , -1.926999999999999894e-02],
      [1.250000000000000000e+00 ,  8.000000000000000444e-01 ,  1.481599999999999931e-02 ,  3.553939999999999877e-01 ,  5.435950000000000504e-01 , -1.317419999999999980e-01 , -1.345599999999999921e-02],
      [2.500000000000000000e+00 ,  8.000000000000000444e-01 ,  1.968999999999999917e-02 ,  4.918299999999999894e-01 ,  7.669930000000000359e-01 , -1.728079999999999894e-01 ,  3.756999999999999923e-03],
      [3.750000000000000000e+00 ,  8.000000000000000444e-01 ,  2.785599999999999882e-02 ,  6.324319999999999942e-01 ,  9.919249999999999456e-01 , -2.077100000000000057e-01 ,  3.159800000000000109e-02],
      [5.000000000000000000e+00 ,  8.000000000000000444e-01 ,  4.394300000000000289e-02 ,  7.650689999999999991e-01 ,  1.188355999999999968e+00 , -2.332680000000000031e-01 ,  5.645000000000000018e-02],
      [0.000000000000000000e+00 ,  8.299999999999999600e-01 ,  1.186100000000000002e-02 ,  2.232899999999999885e-01 ,  3.261100000000000110e-01 , -9.028400000000000314e-02 , -1.806500000000000120e-02],
      [1.000000000000000000e+00 ,  8.299999999999999600e-01 ,  1.444900000000000004e-02 ,  3.383419999999999761e-01 ,  5.161710000000000464e-01 , -1.279530000000000112e-01 , -1.402400000000000001e-02],
      [2.000000000000000000e+00 ,  8.299999999999999600e-01 ,  1.836799999999999891e-02 ,  4.554270000000000262e-01 ,  7.082190000000000429e-01 , -1.642339999999999911e-01 , -1.793000000000000106e-03],
      [3.000000000000000000e+00 ,  8.299999999999999600e-01 ,  2.466899999999999996e-02 ,  5.798410000000000508e-01 ,  9.088819999999999677e-01 , -2.004589999999999983e-01 ,  1.892900000000000138e-02],
      [4.000000000000000000e+00 ,  8.299999999999999600e-01 ,  3.700400000000000217e-02 ,  7.012720000000000065e-01 ,  1.097366000000000064e+00 , -2.362420000000000075e-01 ,  3.750699999999999867e-02],
      [0.000000000000000000e+00 ,  8.599999999999999867e-01 ,  1.224300000000000041e-02 ,  2.278100000000000125e-01 ,  3.342720000000000136e-01 , -9.307600000000000595e-02 , -1.608400000000000107e-02],
      [1.000000000000000000e+00 ,  8.599999999999999867e-01 ,  1.540700000000000056e-02 ,  3.551839999999999997e-01 ,  5.433130000000000459e-01 , -1.364730000000000110e-01 , -1.162200000000000039e-02],
      [2.000000000000000000e+00 ,  8.599999999999999867e-01 ,  2.122699999999999934e-02 ,  4.854620000000000046e-01 ,  7.552919999999999634e-01 , -1.817850000000000021e-01 ,  1.070999999999999903e-03],
      [3.000000000000000000e+00 ,  8.599999999999999867e-01 ,  3.178899999999999781e-02 ,  6.081849999999999756e-01 ,  9.510380000000000500e-01 , -2.252020000000000133e-01 ,  1.540799999999999982e-02],
      [4.000000000000000000e+00 ,  8.599999999999999867e-01 ,  4.744199999999999806e-02 ,  6.846989999999999466e-01 ,  1.042564000000000046e+00 , -2.333600000000000119e-01 ,  2.035400000000000056e-02],
  ])
  
  
  def get_rans_crm_wing():
      # data structure:
      # alpha, mach, cd, cl, cmx, cmy, cmz
  
      deg2rad = np.pi / 180.
  
      xt = np.array(raw[:, 0:2])
      yt = np.array(raw[:, 2:4])
      xlimits = np.array([
          [-3., 10.],
          [0.4, 0.90],
      ])
  
      xt[:, 0] *= deg2rad
      xlimits[0, :] *= deg2rad
  
      return xt, yt, xlimits
  
  
  def plot_rans_crm_wing(xt, yt, limits, interp):
      import numpy as np
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
  
      rad2deg = 180. / np.pi
  
      num = 500
      num_a = 50
      num_M = 50
  
      x = np.zeros((num, 2))
      colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
  
      nrow = 3
      ncol = 2
  
      plt.close()
      plt.figure(figsize=(15, 15))
  
      # -----------------------------------------------------------------------------
  
      mach_numbers = [0.45, 0.68, 0.80, 0.86]
      legend_entries = []
  
      alpha_sweep = np.linspace(0., 8., num)
  
      for ind, mach in enumerate(mach_numbers):
          x[:, 0] = alpha_sweep / rad2deg
          x[:, 1] = mach
          CD = interp.predict_values(x)[:, 0]
          CL = interp.predict_values(x)[:, 1]
  
          plt.subplot(nrow, ncol, 1)
  
          mask = np.abs(xt[:, 1] - mach) < 1e-10
          plt.plot(xt[mask, 0] * rad2deg, yt[mask, 0], 'o' + colors[ind])
          plt.plot(alpha_sweep, CD, colors[ind])
  
          plt.subplot(nrow, ncol, 2)
  
          mask = np.abs(xt[:, 1] - mach) < 1e-10
          plt.plot(xt[mask, 0] * rad2deg, yt[mask, 1], 'o' + colors[ind])
          plt.plot(alpha_sweep, CL, colors[ind])
  
          legend_entries.append('M={}'.format(mach))
          legend_entries.append('exact')
  
      plt.subplot(nrow, ncol, 1)
      plt.xlabel('alpha (deg)')
      plt.ylabel('CD')
      plt.legend(legend_entries)
  
      plt.subplot(nrow, ncol, 2)
      plt.xlabel('alpha (deg)')
      plt.ylabel('CL')
      plt.legend(legend_entries)
  
      # -----------------------------------------------------------------------------
  
      alphas = [2., 4., 6.]
      legend_entries = []
  
      mach_sweep = np.linspace(0.45, 0.86, num)
  
      for ind, alpha in enumerate(alphas):
          x[:, 0] = alpha / rad2deg
          x[:, 1] = mach_sweep
          CD = interp.predict_values(x)[:, 0]
          CL = interp.predict_values(x)[:, 1]
  
          plt.subplot(nrow, ncol, 3)
          plt.plot(mach_sweep, CD, colors[ind])
  
          plt.subplot(nrow, ncol, 4)
          plt.plot(mach_sweep, CL, colors[ind])
  
          legend_entries.append('alpha={}'.format(alpha))
  
      plt.subplot(nrow, ncol, 3)
      plt.xlabel('Mach number')
      plt.ylabel('CD')
      plt.legend(legend_entries)
  
      plt.subplot(nrow, ncol, 4)
      plt.xlabel('Mach number')
      plt.ylabel('CL')
      plt.legend(legend_entries)
  
      # -----------------------------------------------------------------------------
  
      x = np.zeros((num_a, num_M, 2))
      x[:, :, 0] = np.outer(np.linspace(0., 8., num_a), np.ones(num_M)) / rad2deg
      x[:, :, 1] = np.outer(np.ones(num_a), np.linspace(0.45, 0.86, num_M))
      CD = interp.predict_values(x.reshape((num_a * num_M, 2)))[:, 0].reshape((num_a, num_M))
      CL = interp.predict_values(x.reshape((num_a * num_M, 2)))[:, 1].reshape((num_a, num_M))
  
      plt.subplot(nrow, ncol, 5)
      plt.plot(xt[:, 1], xt[:, 0] * rad2deg, 'o')
      plt.contour(x[:, :, 1], x[:, :, 0] * rad2deg, CD, 20)
      plt.pcolormesh(x[:, :, 1], x[:, :, 0] * rad2deg, CD, cmap = plt.get_cmap('rainbow'))
      plt.xlabel('Mach number')
      plt.ylabel('alpha (deg)')
      plt.title('CD')
      plt.colorbar()
  
      plt.subplot(nrow, ncol, 6)
      plt.plot(xt[:, 1], xt[:, 0] * rad2deg, 'o')
      plt.contour(x[:, :, 1], x[:, :, 0] * rad2deg, CL, 20)
      plt.pcolormesh(x[:, :, 1], x[:, :, 0] * rad2deg, CL, cmap = plt.get_cmap('rainbow'))
      plt.xlabel('Mach number')
      plt.ylabel('alpha (deg)')
      plt.title('CL')
      plt.colorbar()
  
      plt.show()
  

RMTB
----

.. code-block:: python

  from smt.surrogate_models import RMTB
  from smt.examples.rans_crm_wing.rans_crm_wing import get_rans_crm_wing, plot_rans_crm_wing
  
  xt, yt, xlimits = get_rans_crm_wing()
  
  interp = RMTB(num_ctrl_pts=20, xlimits=xlimits, nonlinear_maxiter=100, energy_weight=1e-12)
  interp.set_training_values(xt, yt)
  interp.train()
  
  plot_rans_crm_wing(xt, yt, xlimits, interp)
  
::

  ___________________________________________________________________________
     
                                     RMTB
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 35
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Pre-computing matrices ...
           Computing dof2coeff ...
           Computing dof2coeff - done. Time (sec):  0.0000031
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0003679
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0043740
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0004060
        Pre-computing matrices - done. Time (sec):  0.0052021
        Solving for degrees of freedom ...
           Solving initial startup problem (n=400) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.429150220e-02 1.114942861e-02
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.190705479e-08 1.793043866e-10
              Solving for output 0 - done. Time (sec):  0.0089231
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.955493282e+00 4.799845498e+00
                 Iteration (num., iy, grad. norm, func.) :   0   1 9.305384587e-07 4.567778222e-08
              Solving for output 1 - done. Time (sec):  0.0087180
           Solving initial startup problem (n=400) - done. Time (sec):  0.0177021
           Solving nonlinear problem (n=400) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 6.652643595e-09 1.793037365e-10
                 Iteration (num., iy, grad. norm, func.) :   0   0 5.849550610e-09 1.703948816e-10
                 Iteration (num., iy, grad. norm, func.) :   1   0 3.020048141e-08 1.031960180e-10
                 Iteration (num., iy, grad. norm, func.) :   2   0 1.118995039e-08 2.491680542e-11
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.515519434e-09 1.045856653e-11
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.437729168e-09 9.448359444e-12
                 Iteration (num., iy, grad. norm, func.) :   5   0 6.972963690e-10 7.402298884e-12
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.047037695e-10 6.532866174e-12
                 Iteration (num., iy, grad. norm, func.) :   7   0 4.278608007e-11 6.262495725e-12
                 Iteration (num., iy, grad. norm, func.) :   8   0 2.563618203e-11 6.261553276e-12
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.667337829e-11 6.260601900e-12
                 Iteration (num., iy, grad. norm, func.) :  10   0 9.714039915e-12 6.260265201e-12
                 Iteration (num., iy, grad. norm, func.) :  11   0 3.314878677e-12 6.256671174e-12
                 Iteration (num., iy, grad. norm, func.) :  12   0 7.167083031e-13 6.255690441e-12
              Solving for output 0 - done. Time (sec):  0.1148291
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 9.728345943e-08 4.567643711e-08
                 Iteration (num., iy, grad. norm, func.) :   0   1 9.337082620e-08 4.538211616e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 2.886043606e-06 3.235600341e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 8.533006318e-07 4.616064024e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 3.216546950e-07 2.307416728e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 3.156322657e-07 2.238089688e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 1.052616496e-07 7.302994308e-10
                 Iteration (num., iy, grad. norm, func.) :   6   1 5.199173368e-08 5.755259354e-10
                 Iteration (num., iy, grad. norm, func.) :   7   1 1.508588567e-08 5.049283664e-10
                 Iteration (num., iy, grad. norm, func.) :   8   1 8.881860761e-09 3.811496561e-10
                 Iteration (num., iy, grad. norm, func.) :   9   1 5.285490688e-09 3.025159178e-10
                 Iteration (num., iy, grad. norm, func.) :  10   1 1.404487184e-09 2.730620615e-10
                 Iteration (num., iy, grad. norm, func.) :  11   1 8.186126007e-10 2.725545391e-10
                 Iteration (num., iy, grad. norm, func.) :  12   1 4.496744504e-10 2.724158427e-10
                 Iteration (num., iy, grad. norm, func.) :  13   1 4.308883634e-10 2.723151638e-10
                 Iteration (num., iy, grad. norm, func.) :  14   1 9.560586878e-10 2.723036622e-10
                 Iteration (num., iy, grad. norm, func.) :  15   1 2.277664919e-10 2.715444113e-10
                 Iteration (num., iy, grad. norm, func.) :  16   1 6.294915055e-11 2.714190487e-10
                 Iteration (num., iy, grad. norm, func.) :  17   1 6.231468691e-11 2.714154864e-10
                 Iteration (num., iy, grad. norm, func.) :  18   1 4.018887519e-11 2.714051799e-10
                 Iteration (num., iy, grad. norm, func.) :  19   1 1.853379466e-11 2.713730777e-10
                 Iteration (num., iy, grad. norm, func.) :  20   1 2.638239660e-11 2.713513914e-10
                 Iteration (num., iy, grad. norm, func.) :  21   1 1.091254506e-11 2.713506877e-10
                 Iteration (num., iy, grad. norm, func.) :  22   1 9.671762748e-12 2.713494300e-10
                 Iteration (num., iy, grad. norm, func.) :  23   1 5.346363869e-12 2.713468874e-10
                 Iteration (num., iy, grad. norm, func.) :  24   1 5.710962220e-12 2.713457960e-10
                 Iteration (num., iy, grad. norm, func.) :  25   1 4.612943372e-12 2.713455978e-10
                 Iteration (num., iy, grad. norm, func.) :  26   1 6.788718160e-12 2.713452805e-10
                 Iteration (num., iy, grad. norm, func.) :  27   1 1.360254705e-12 2.713450624e-10
                 Iteration (num., iy, grad. norm, func.) :  28   1 1.107836494e-12 2.713450491e-10
                 Iteration (num., iy, grad. norm, func.) :  29   1 1.623738342e-12 2.713449869e-10
                 Iteration (num., iy, grad. norm, func.) :  30   1 5.434052411e-13 2.713449484e-10
              Solving for output 1 - done. Time (sec):  0.2859800
           Solving nonlinear problem (n=400) - done. Time (sec):  0.4008608
        Solving for degrees of freedom - done. Time (sec):  0.4186170
     Training - done. Time (sec):  0.4242423
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0004890
     
     Prediction time/pt. (sec) :  0.0000010
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0004451
     
     Prediction time/pt. (sec) :  0.0000009
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005250
     
     Prediction time/pt. (sec) :  0.0000010
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0004511
     
     Prediction time/pt. (sec) :  0.0000009
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005019
     
     Prediction time/pt. (sec) :  0.0000010
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0004570
     
     Prediction time/pt. (sec) :  0.0000009
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005431
     
     Prediction time/pt. (sec) :  0.0000011
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0004840
     
     Prediction time/pt. (sec) :  0.0000010
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005202
     
     Prediction time/pt. (sec) :  0.0000010
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0004470
     
     Prediction time/pt. (sec) :  0.0000009
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005622
     
     Prediction time/pt. (sec) :  0.0000011
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0004501
     
     Prediction time/pt. (sec) :  0.0000009
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007071
     
     Prediction time/pt. (sec) :  0.0000014
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005000
     
     Prediction time/pt. (sec) :  0.0000010
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013919
     
     Prediction time/pt. (sec) :  0.0000006
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012252
     
     Prediction time/pt. (sec) :  0.0000005
     
  
.. figure:: rans_crm_wing.png
  :scale: 60 %
  :align: center

RMTC
----

.. code-block:: python

  from smt.surrogate_models import RMTC
  from smt.examples.rans_crm_wing.rans_crm_wing import get_rans_crm_wing, plot_rans_crm_wing
  
  xt, yt, xlimits = get_rans_crm_wing()
  
  interp = RMTC(num_elements=20, xlimits=xlimits, nonlinear_maxiter=100, energy_weight=1e-10)
  interp.set_training_values(xt, yt)
  interp.train()
  
  plot_rans_crm_wing(xt, yt, xlimits, interp)
  
::

  ___________________________________________________________________________
     
                                     RMTC
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 35
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Pre-computing matrices ...
           Computing dof2coeff ...
           Computing dof2coeff - done. Time (sec):  0.0033829
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0003350
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0118701
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0008340
        Pre-computing matrices - done. Time (sec):  0.0165062
        Solving for degrees of freedom ...
           Solving initial startup problem (n=1764) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.279175539e-01 1.114942861e-02
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.100030910e-05 2.198207064e-08
              Solving for output 0 - done. Time (sec):  0.0203757
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 2.653045755e+00 4.799845498e+00
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.207814299e-04 6.117234264e-06
              Solving for output 1 - done. Time (sec):  0.0203140
           Solving initial startup problem (n=1764) - done. Time (sec):  0.0407798
           Solving nonlinear problem (n=1764) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 8.429448954e-07 2.189151457e-08
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.383447488e-07 1.739738698e-08
                 Iteration (num., iy, grad. norm, func.) :   1   0 3.530444505e-07 3.265182162e-09
                 Iteration (num., iy, grad. norm, func.) :   2   0 1.177503648e-07 1.050570789e-09
                 Iteration (num., iy, grad. norm, func.) :   3   0 6.359395346e-08 5.341289379e-10
                 Iteration (num., iy, grad. norm, func.) :   4   0 3.402795059e-08 4.107220720e-10
                 Iteration (num., iy, grad. norm, func.) :   5   0 2.254406671e-08 3.755943389e-10
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.949657116e-08 3.754924624e-10
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.483902867e-08 3.664669055e-10
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.687187082e-08 3.644310217e-10
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.437741686e-08 3.402842318e-10
                 Iteration (num., iy, grad. norm, func.) :  10   0 8.455286334e-09 3.089193365e-10
                 Iteration (num., iy, grad. norm, func.) :  11   0 2.831612717e-09 2.906169561e-10
                 Iteration (num., iy, grad. norm, func.) :  12   0 2.298848598e-09 2.894217952e-10
                 Iteration (num., iy, grad. norm, func.) :  13   0 2.298848594e-09 2.894217952e-10
                 Iteration (num., iy, grad. norm, func.) :  14   0 2.298848594e-09 2.894217952e-10
                 Iteration (num., iy, grad. norm, func.) :  15   0 3.782427426e-09 2.884421666e-10
                 Iteration (num., iy, grad. norm, func.) :  16   0 1.057557139e-09 2.873111720e-10
                 Iteration (num., iy, grad. norm, func.) :  17   0 1.374903118e-09 2.870605201e-10
                 Iteration (num., iy, grad. norm, func.) :  18   0 9.935172696e-10 2.869697391e-10
                 Iteration (num., iy, grad. norm, func.) :  19   0 9.081091180e-10 2.869525725e-10
                 Iteration (num., iy, grad. norm, func.) :  20   0 9.093909859e-10 2.869065444e-10
                 Iteration (num., iy, grad. norm, func.) :  21   0 1.260539881e-09 2.867589584e-10
                 Iteration (num., iy, grad. norm, func.) :  22   0 6.797994146e-10 2.866334737e-10
                 Iteration (num., iy, grad. norm, func.) :  23   0 3.633671112e-10 2.865681775e-10
                 Iteration (num., iy, grad. norm, func.) :  24   0 3.133470071e-10 2.865649406e-10
                 Iteration (num., iy, grad. norm, func.) :  25   0 4.575456293e-10 2.865619950e-10
                 Iteration (num., iy, grad. norm, func.) :  26   0 3.691928446e-10 2.865532154e-10
                 Iteration (num., iy, grad. norm, func.) :  27   0 5.697858816e-10 2.865435462e-10
                 Iteration (num., iy, grad. norm, func.) :  28   0 3.000559676e-10 2.865282893e-10
                 Iteration (num., iy, grad. norm, func.) :  29   0 3.687835789e-10 2.865200812e-10
                 Iteration (num., iy, grad. norm, func.) :  30   0 1.753823829e-10 2.865129931e-10
                 Iteration (num., iy, grad. norm, func.) :  31   0 2.489293002e-10 2.865043719e-10
                 Iteration (num., iy, grad. norm, func.) :  32   0 7.363220087e-11 2.864970631e-10
                 Iteration (num., iy, grad. norm, func.) :  33   0 5.389234516e-11 2.864970135e-10
                 Iteration (num., iy, grad. norm, func.) :  34   0 7.411684730e-11 2.864965217e-10
                 Iteration (num., iy, grad. norm, func.) :  35   0 1.023462658e-10 2.864956025e-10
                 Iteration (num., iy, grad. norm, func.) :  36   0 9.411789760e-11 2.864950533e-10
                 Iteration (num., iy, grad. norm, func.) :  37   0 7.442365830e-11 2.864946735e-10
                 Iteration (num., iy, grad. norm, func.) :  38   0 9.876699892e-11 2.864945963e-10
                 Iteration (num., iy, grad. norm, func.) :  39   0 5.447084430e-11 2.864941303e-10
                 Iteration (num., iy, grad. norm, func.) :  40   0 6.559094646e-11 2.864935572e-10
                 Iteration (num., iy, grad. norm, func.) :  41   0 2.444248854e-11 2.864929592e-10
                 Iteration (num., iy, grad. norm, func.) :  42   0 3.421370587e-11 2.864929249e-10
                 Iteration (num., iy, grad. norm, func.) :  43   0 2.458546948e-11 2.864928725e-10
                 Iteration (num., iy, grad. norm, func.) :  44   0 3.853334986e-11 2.864928017e-10
                 Iteration (num., iy, grad. norm, func.) :  45   0 2.420358359e-11 2.864926759e-10
                 Iteration (num., iy, grad. norm, func.) :  46   0 2.943826714e-11 2.864926236e-10
                 Iteration (num., iy, grad. norm, func.) :  47   0 1.830936268e-11 2.864925921e-10
                 Iteration (num., iy, grad. norm, func.) :  48   0 2.556701849e-11 2.864925622e-10
                 Iteration (num., iy, grad. norm, func.) :  49   0 1.075462329e-11 2.864925170e-10
                 Iteration (num., iy, grad. norm, func.) :  50   0 1.383237043e-11 2.864925020e-10
                 Iteration (num., iy, grad. norm, func.) :  51   0 1.164299243e-11 2.864924848e-10
                 Iteration (num., iy, grad. norm, func.) :  52   0 1.815693339e-11 2.864924627e-10
                 Iteration (num., iy, grad. norm, func.) :  53   0 5.233745611e-12 2.864924423e-10
                 Iteration (num., iy, grad. norm, func.) :  54   0 1.044172372e-11 2.864924409e-10
                 Iteration (num., iy, grad. norm, func.) :  55   0 6.645968011e-12 2.864924386e-10
                 Iteration (num., iy, grad. norm, func.) :  56   0 8.941457068e-12 2.864924366e-10
                 Iteration (num., iy, grad. norm, func.) :  57   0 4.693420943e-12 2.864924304e-10
                 Iteration (num., iy, grad. norm, func.) :  58   0 5.258752918e-12 2.864924264e-10
                 Iteration (num., iy, grad. norm, func.) :  59   0 3.107228869e-12 2.864924229e-10
                 Iteration (num., iy, grad. norm, func.) :  60   0 4.176070800e-12 2.864924212e-10
                 Iteration (num., iy, grad. norm, func.) :  61   0 2.483609103e-12 2.864924200e-10
                 Iteration (num., iy, grad. norm, func.) :  62   0 3.352836754e-12 2.864924196e-10
                 Iteration (num., iy, grad. norm, func.) :  63   0 2.115431579e-12 2.864924187e-10
                 Iteration (num., iy, grad. norm, func.) :  64   0 2.852586385e-12 2.864924180e-10
                 Iteration (num., iy, grad. norm, func.) :  65   0 1.547869431e-12 2.864924172e-10
                 Iteration (num., iy, grad. norm, func.) :  66   0 1.981061155e-12 2.864924168e-10
                 Iteration (num., iy, grad. norm, func.) :  67   0 1.346810106e-12 2.864924164e-10
                 Iteration (num., iy, grad. norm, func.) :  68   0 1.863909571e-12 2.864924163e-10
                 Iteration (num., iy, grad. norm, func.) :  69   0 1.106608266e-12 2.864924161e-10
                 Iteration (num., iy, grad. norm, func.) :  70   0 1.109367469e-12 2.864924160e-10
                 Iteration (num., iy, grad. norm, func.) :  71   0 7.870538430e-13 2.864924158e-10
              Solving for output 0 - done. Time (sec):  1.4013419
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.400349919e-05 6.107787296e-06
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.352579123e-05 5.875347456e-06
                 Iteration (num., iy, grad. norm, func.) :   1   1 1.501537773e-05 7.944375821e-07
                 Iteration (num., iy, grad. norm, func.) :   2   1 1.500776695e-05 2.989993077e-07
                 Iteration (num., iy, grad. norm, func.) :   3   1 4.779541206e-06 1.111517625e-07
                 Iteration (num., iy, grad. norm, func.) :   4   1 3.857354797e-06 7.291168344e-08
                 Iteration (num., iy, grad. norm, func.) :   5   1 3.035357589e-06 5.612895860e-08
                 Iteration (num., iy, grad. norm, func.) :   6   1 9.323805017e-07 2.528290313e-08
                 Iteration (num., iy, grad. norm, func.) :   7   1 7.269980822e-07 2.430391947e-08
                 Iteration (num., iy, grad. norm, func.) :   8   1 5.630956438e-07 2.417341048e-08
                 Iteration (num., iy, grad. norm, func.) :   9   1 3.206611473e-07 2.217555019e-08
                 Iteration (num., iy, grad. norm, func.) :  10   1 8.569285321e-08 1.853280233e-08
                 Iteration (num., iy, grad. norm, func.) :  11   1 8.548469578e-08 1.604155829e-08
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.875255730e-08 1.460356669e-08
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.201969328e-08 1.454032308e-08
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.201969328e-08 1.454032308e-08
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.201969328e-08 1.454032308e-08
                 Iteration (num., iy, grad. norm, func.) :  16   1 1.428814369e-08 1.452513496e-08
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.162916854e-08 1.450995898e-08
                 Iteration (num., iy, grad. norm, func.) :  18   1 1.753757644e-08 1.449519571e-08
                 Iteration (num., iy, grad. norm, func.) :  19   1 3.068211340e-09 1.447319260e-08
                 Iteration (num., iy, grad. norm, func.) :  20   1 3.511212322e-09 1.447244704e-08
                 Iteration (num., iy, grad. norm, func.) :  21   1 5.065392233e-09 1.447210114e-08
                 Iteration (num., iy, grad. norm, func.) :  22   1 4.234122344e-09 1.447197915e-08
                 Iteration (num., iy, grad. norm, func.) :  23   1 6.576922309e-09 1.447033147e-08
                 Iteration (num., iy, grad. norm, func.) :  24   1 2.978190644e-09 1.446781783e-08
                 Iteration (num., iy, grad. norm, func.) :  25   1 3.613729658e-09 1.446669747e-08
                 Iteration (num., iy, grad. norm, func.) :  26   1 1.865903444e-09 1.446588369e-08
                 Iteration (num., iy, grad. norm, func.) :  27   1 3.110409518e-09 1.446458243e-08
                 Iteration (num., iy, grad. norm, func.) :  28   1 6.363917785e-10 1.446405612e-08
                 Iteration (num., iy, grad. norm, func.) :  29   1 6.363915076e-10 1.446405612e-08
                 Iteration (num., iy, grad. norm, func.) :  30   1 9.392018781e-10 1.446405085e-08
                 Iteration (num., iy, grad. norm, func.) :  31   1 1.163097651e-09 1.446390004e-08
                 Iteration (num., iy, grad. norm, func.) :  32   1 7.852152889e-10 1.446373811e-08
                 Iteration (num., iy, grad. norm, func.) :  33   1 5.887769343e-10 1.446366832e-08
                 Iteration (num., iy, grad. norm, func.) :  34   1 4.599413093e-10 1.446365443e-08
                 Iteration (num., iy, grad. norm, func.) :  35   1 6.433078256e-10 1.446365233e-08
                 Iteration (num., iy, grad. norm, func.) :  36   1 4.455729393e-10 1.446364229e-08
                 Iteration (num., iy, grad. norm, func.) :  37   1 5.232475384e-10 1.446362765e-08
                 Iteration (num., iy, grad. norm, func.) :  38   1 4.650183368e-10 1.446360291e-08
                 Iteration (num., iy, grad. norm, func.) :  39   1 1.627087301e-10 1.446358056e-08
                 Iteration (num., iy, grad. norm, func.) :  40   1 2.265437919e-10 1.446357715e-08
                 Iteration (num., iy, grad. norm, func.) :  41   1 1.626683795e-10 1.446357548e-08
                 Iteration (num., iy, grad. norm, func.) :  42   1 2.508786641e-10 1.446357327e-08
                 Iteration (num., iy, grad. norm, func.) :  43   1 1.300695856e-10 1.446356869e-08
                 Iteration (num., iy, grad. norm, func.) :  44   1 2.001504124e-10 1.446356622e-08
                 Iteration (num., iy, grad. norm, func.) :  45   1 8.341859078e-11 1.446356365e-08
                 Iteration (num., iy, grad. norm, func.) :  46   1 1.077507305e-10 1.446356315e-08
                 Iteration (num., iy, grad. norm, func.) :  47   1 8.396499958e-11 1.446356283e-08
                 Iteration (num., iy, grad. norm, func.) :  48   1 1.371454829e-10 1.446356083e-08
                 Iteration (num., iy, grad. norm, func.) :  49   1 1.847401542e-11 1.446355955e-08
                 Iteration (num., iy, grad. norm, func.) :  50   1 1.846967329e-11 1.446355955e-08
                 Iteration (num., iy, grad. norm, func.) :  51   1 2.655936552e-11 1.446355953e-08
                 Iteration (num., iy, grad. norm, func.) :  52   1 3.002981479e-11 1.446355947e-08
                 Iteration (num., iy, grad. norm, func.) :  53   1 2.959674188e-11 1.446355941e-08
                 Iteration (num., iy, grad. norm, func.) :  54   1 2.468046322e-11 1.446355938e-08
                 Iteration (num., iy, grad. norm, func.) :  55   1 2.505122646e-11 1.446355931e-08
                 Iteration (num., iy, grad. norm, func.) :  56   1 1.331008825e-11 1.446355926e-08
                 Iteration (num., iy, grad. norm, func.) :  57   1 1.635443193e-11 1.446355923e-08
                 Iteration (num., iy, grad. norm, func.) :  58   1 1.068366534e-11 1.446355920e-08
                 Iteration (num., iy, grad. norm, func.) :  59   1 1.271980671e-11 1.446355920e-08
                 Iteration (num., iy, grad. norm, func.) :  60   1 1.094073833e-11 1.446355919e-08
                 Iteration (num., iy, grad. norm, func.) :  61   1 1.469239859e-11 1.446355918e-08
                 Iteration (num., iy, grad. norm, func.) :  62   1 7.265078730e-12 1.446355917e-08
                 Iteration (num., iy, grad. norm, func.) :  63   1 6.581576596e-12 1.446355916e-08
                 Iteration (num., iy, grad. norm, func.) :  64   1 5.753360418e-12 1.446355916e-08
                 Iteration (num., iy, grad. norm, func.) :  65   1 6.734399946e-12 1.446355916e-08
                 Iteration (num., iy, grad. norm, func.) :  66   1 4.345267021e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  67   1 6.826019896e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  68   1 2.453398194e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  69   1 3.707296274e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  70   1 2.068273743e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  71   1 1.663351429e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  72   1 1.836482500e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  73   1 2.683725701e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  74   1 1.126064165e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  75   1 2.077125676e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  76   1 6.966048244e-13 1.446355915e-08
              Solving for output 1 - done. Time (sec):  1.9143078
           Solving nonlinear problem (n=1764) - done. Time (sec):  3.3157179
        Solving for degrees of freedom - done. Time (sec):  3.3565691
     Training - done. Time (sec):  3.3738129
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007412
     
     Prediction time/pt. (sec) :  0.0000015
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006618
     
     Prediction time/pt. (sec) :  0.0000013
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0008941
     
     Prediction time/pt. (sec) :  0.0000018
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007570
     
     Prediction time/pt. (sec) :  0.0000015
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007889
     
     Prediction time/pt. (sec) :  0.0000016
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007951
     
     Prediction time/pt. (sec) :  0.0000016
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0008152
     
     Prediction time/pt. (sec) :  0.0000016
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007339
     
     Prediction time/pt. (sec) :  0.0000015
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0008512
     
     Prediction time/pt. (sec) :  0.0000017
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007653
     
     Prediction time/pt. (sec) :  0.0000015
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0008521
     
     Prediction time/pt. (sec) :  0.0000017
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007658
     
     Prediction time/pt. (sec) :  0.0000015
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007820
     
     Prediction time/pt. (sec) :  0.0000016
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007558
     
     Prediction time/pt. (sec) :  0.0000015
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0028410
     
     Prediction time/pt. (sec) :  0.0000011
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027227
     
     Prediction time/pt. (sec) :  0.0000011
     
  
.. figure:: rans_crm_wing.png
  :scale: 60 %
  :align: center
