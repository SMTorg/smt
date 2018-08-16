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
           Computing dof2coeff - done. Time (sec):  0.0000019
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0004699
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0057001
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0006211
        Pre-computing matrices - done. Time (sec):  0.0068741
        Solving for degrees of freedom ...
           Solving initial startup problem (n=400) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.429150220e-02 1.114942861e-02
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.126996280e-08 1.793039622e-10
              Solving for output 0 - done. Time (sec):  0.0158942
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.955493282e+00 4.799845498e+00
                 Iteration (num., iy, grad. norm, func.) :   0   1 2.697466889e-07 4.567649850e-08
              Solving for output 1 - done. Time (sec):  0.0157371
           Solving initial startup problem (n=400) - done. Time (sec):  0.0317152
           Solving nonlinear problem (n=400) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 6.652693068e-09 1.793037801e-10
                 Iteration (num., iy, grad. norm, func.) :   0   0 5.849617718e-09 1.703950713e-10
                 Iteration (num., iy, grad. norm, func.) :   1   0 3.025566935e-08 1.033343258e-10
                 Iteration (num., iy, grad. norm, func.) :   2   0 1.125487407e-08 2.504487682e-11
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.535709346e-09 1.049407146e-11
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.494883650e-09 9.524089801e-12
                 Iteration (num., iy, grad. norm, func.) :   5   0 7.133349034e-10 7.416110278e-12
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.004404072e-10 6.533121174e-12
                 Iteration (num., iy, grad. norm, func.) :   7   0 4.190689804e-11 6.262358032e-12
                 Iteration (num., iy, grad. norm, func.) :   8   0 2.482185245e-11 6.261715105e-12
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.657786466e-11 6.260755557e-12
                 Iteration (num., iy, grad. norm, func.) :  10   0 9.761577910e-12 6.260204240e-12
                 Iteration (num., iy, grad. norm, func.) :  11   0 2.795276212e-12 6.256591279e-12
                 Iteration (num., iy, grad. norm, func.) :  12   0 5.997502139e-13 6.255690394e-12
              Solving for output 0 - done. Time (sec):  0.2036891
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 9.729453645e-08 4.567641275e-08
                 Iteration (num., iy, grad. norm, func.) :   0   1 9.338312477e-08 4.538212969e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 2.886091534e-06 3.234109162e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 8.559781824e-07 4.637978549e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 2.823691609e-07 2.049077806e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 2.723236688e-07 1.947641893e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 1.792928010e-07 1.210775835e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 1.181658522e-07 7.711278799e-10
                 Iteration (num., iy, grad. norm, func.) :   7   1 3.467121432e-08 4.634059415e-10
                 Iteration (num., iy, grad. norm, func.) :   8   1 9.984565229e-09 4.213082093e-10
                 Iteration (num., iy, grad. norm, func.) :   9   1 3.068167164e-09 3.319977044e-10
                 Iteration (num., iy, grad. norm, func.) :  10   1 1.186604737e-09 2.807275088e-10
                 Iteration (num., iy, grad. norm, func.) :  11   1 3.906767960e-10 2.731330653e-10
                 Iteration (num., iy, grad. norm, func.) :  12   1 3.614894546e-10 2.730008762e-10
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.215479619e-10 2.721116067e-10
                 Iteration (num., iy, grad. norm, func.) :  14   1 5.702469698e-11 2.715757739e-10
                 Iteration (num., iy, grad. norm, func.) :  15   1 3.672983378e-11 2.714454914e-10
                 Iteration (num., iy, grad. norm, func.) :  16   1 6.502221017e-11 2.714407840e-10
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.743600885e-11 2.713891416e-10
                 Iteration (num., iy, grad. norm, func.) :  18   1 4.788887986e-11 2.713889196e-10
                 Iteration (num., iy, grad. norm, func.) :  19   1 1.196201880e-11 2.713646665e-10
                 Iteration (num., iy, grad. norm, func.) :  20   1 1.562648133e-11 2.713514888e-10
                 Iteration (num., iy, grad. norm, func.) :  21   1 6.261087902e-12 2.713491651e-10
                 Iteration (num., iy, grad. norm, func.) :  22   1 1.400320341e-11 2.713490729e-10
                 Iteration (num., iy, grad. norm, func.) :  23   1 2.542300839e-12 2.713461617e-10
                 Iteration (num., iy, grad. norm, func.) :  24   1 3.884621068e-12 2.713457860e-10
                 Iteration (num., iy, grad. norm, func.) :  25   1 6.967485435e-12 2.713455447e-10
                 Iteration (num., iy, grad. norm, func.) :  26   1 3.566001156e-12 2.713452990e-10
                 Iteration (num., iy, grad. norm, func.) :  27   1 3.325579111e-12 2.713451178e-10
                 Iteration (num., iy, grad. norm, func.) :  28   1 1.877850418e-12 2.713450513e-10
                 Iteration (num., iy, grad. norm, func.) :  29   1 1.435419086e-12 2.713449988e-10
                 Iteration (num., iy, grad. norm, func.) :  30   1 1.996402793e-12 2.713449648e-10
                 Iteration (num., iy, grad. norm, func.) :  31   1 7.066346220e-13 2.713449432e-10
              Solving for output 1 - done. Time (sec):  0.5002341
           Solving nonlinear problem (n=400) - done. Time (sec):  0.7039950
        Solving for degrees of freedom - done. Time (sec):  0.7357800
     Training - done. Time (sec):  0.7432971
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006590
     
     Prediction time/pt. (sec) :  0.0000013
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006061
     
     Prediction time/pt. (sec) :  0.0000012
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006781
     
     Prediction time/pt. (sec) :  0.0000014
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006139
     
     Prediction time/pt. (sec) :  0.0000012
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006499
     
     Prediction time/pt. (sec) :  0.0000013
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006258
     
     Prediction time/pt. (sec) :  0.0000013
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006731
     
     Prediction time/pt. (sec) :  0.0000013
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006201
     
     Prediction time/pt. (sec) :  0.0000012
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006671
     
     Prediction time/pt. (sec) :  0.0000013
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006089
     
     Prediction time/pt. (sec) :  0.0000012
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006590
     
     Prediction time/pt. (sec) :  0.0000013
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006180
     
     Prediction time/pt. (sec) :  0.0000012
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006292
     
     Prediction time/pt. (sec) :  0.0000013
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006158
     
     Prediction time/pt. (sec) :  0.0000012
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0016150
     
     Prediction time/pt. (sec) :  0.0000006
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0015090
     
     Prediction time/pt. (sec) :  0.0000006
     
  
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
           Computing dof2coeff - done. Time (sec):  0.0045240
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0004928
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0148189
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0011470
        Pre-computing matrices - done. Time (sec):  0.0210750
        Solving for degrees of freedom ...
           Solving initial startup problem (n=1764) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.279175539e-01 1.114942861e-02
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.525626130e-05 2.184891017e-08
              Solving for output 0 - done. Time (sec):  0.0348730
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 2.653045755e+00 4.799845498e+00
                 Iteration (num., iy, grad. norm, func.) :   0   1 6.590708234e-05 6.501028429e-06
              Solving for output 1 - done. Time (sec):  0.0353301
           Solving initial startup problem (n=1764) - done. Time (sec):  0.0703139
           Solving nonlinear problem (n=1764) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 8.078914091e-07 2.166447289e-08
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.144203614e-07 1.723325764e-08
                 Iteration (num., iy, grad. norm, func.) :   1   0 3.512848564e-07 3.252718660e-09
                 Iteration (num., iy, grad. norm, func.) :   2   0 1.168240718e-07 1.046750212e-09
                 Iteration (num., iy, grad. norm, func.) :   3   0 6.329288940e-08 5.331255046e-10
                 Iteration (num., iy, grad. norm, func.) :   4   0 3.416970363e-08 4.111249340e-10
                 Iteration (num., iy, grad. norm, func.) :   5   0 2.254659429e-08 3.760797805e-10
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.009632074e-08 3.714401754e-10
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.934766932e-08 3.710416397e-10
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.336142337e-08 3.606158144e-10
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.484071371e-08 3.420934208e-10
                 Iteration (num., iy, grad. norm, func.) :  10   0 7.058192894e-09 3.069629237e-10
                 Iteration (num., iy, grad. norm, func.) :  11   0 2.414987245e-09 2.895245402e-10
                 Iteration (num., iy, grad. norm, func.) :  12   0 1.985215675e-09 2.891910851e-10
                 Iteration (num., iy, grad. norm, func.) :  13   0 2.274383828e-09 2.889768689e-10
                 Iteration (num., iy, grad. norm, func.) :  14   0 1.508023702e-09 2.879190117e-10
                 Iteration (num., iy, grad. norm, func.) :  15   0 1.620847037e-09 2.871365577e-10
                 Iteration (num., iy, grad. norm, func.) :  16   0 5.731149757e-10 2.868379332e-10
                 Iteration (num., iy, grad. norm, func.) :  17   0 5.008888169e-10 2.868165672e-10
                 Iteration (num., iy, grad. norm, func.) :  18   0 8.031942070e-10 2.868034482e-10
                 Iteration (num., iy, grad. norm, func.) :  19   0 9.994476052e-10 2.867800583e-10
                 Iteration (num., iy, grad. norm, func.) :  20   0 7.733739008e-10 2.867410067e-10
                 Iteration (num., iy, grad. norm, func.) :  21   0 9.139102587e-10 2.866789467e-10
                 Iteration (num., iy, grad. norm, func.) :  22   0 3.644138954e-10 2.865887406e-10
                 Iteration (num., iy, grad. norm, func.) :  23   0 4.324586053e-10 2.865771771e-10
                 Iteration (num., iy, grad. norm, func.) :  24   0 5.071061961e-10 2.865753801e-10
                 Iteration (num., iy, grad. norm, func.) :  25   0 5.866543693e-10 2.865685353e-10
                 Iteration (num., iy, grad. norm, func.) :  26   0 3.492546224e-10 2.865532358e-10
                 Iteration (num., iy, grad. norm, func.) :  27   0 4.515879616e-10 2.865340799e-10
                 Iteration (num., iy, grad. norm, func.) :  28   0 1.592983333e-10 2.865166461e-10
                 Iteration (num., iy, grad. norm, func.) :  29   0 2.416735330e-10 2.865116592e-10
                 Iteration (num., iy, grad. norm, func.) :  30   0 2.184128801e-10 2.865105385e-10
                 Iteration (num., iy, grad. norm, func.) :  31   0 2.308349493e-10 2.865091909e-10
                 Iteration (num., iy, grad. norm, func.) :  32   0 2.347464637e-10 2.865035896e-10
                 Iteration (num., iy, grad. norm, func.) :  33   0 7.188318332e-11 2.864965865e-10
                 Iteration (num., iy, grad. norm, func.) :  34   0 5.775926076e-11 2.864960437e-10
                 Iteration (num., iy, grad. norm, func.) :  35   0 7.734861517e-11 2.864955112e-10
                 Iteration (num., iy, grad. norm, func.) :  36   0 8.083117479e-11 2.864949805e-10
                 Iteration (num., iy, grad. norm, func.) :  37   0 7.147073438e-11 2.864932077e-10
                 Iteration (num., iy, grad. norm, func.) :  38   0 3.451763296e-11 2.864929979e-10
                 Iteration (num., iy, grad. norm, func.) :  39   0 3.451762397e-11 2.864929979e-10
                 Iteration (num., iy, grad. norm, func.) :  40   0 3.451762214e-11 2.864929979e-10
                 Iteration (num., iy, grad. norm, func.) :  41   0 4.605470389e-11 2.864928044e-10
                 Iteration (num., iy, grad. norm, func.) :  42   0 6.464587406e-12 2.864925251e-10
                 Iteration (num., iy, grad. norm, func.) :  43   0 7.824747529e-12 2.864924940e-10
                 Iteration (num., iy, grad. norm, func.) :  44   0 1.188992500e-11 2.864924873e-10
                 Iteration (num., iy, grad. norm, func.) :  45   0 1.006451501e-11 2.864924789e-10
                 Iteration (num., iy, grad. norm, func.) :  46   0 3.055319401e-11 2.864924714e-10
                 Iteration (num., iy, grad. norm, func.) :  47   0 8.273331413e-12 2.864924593e-10
                 Iteration (num., iy, grad. norm, func.) :  48   0 9.360617237e-12 2.864924556e-10
                 Iteration (num., iy, grad. norm, func.) :  49   0 7.254676117e-12 2.864924478e-10
                 Iteration (num., iy, grad. norm, func.) :  50   0 1.110207770e-11 2.864924384e-10
                 Iteration (num., iy, grad. norm, func.) :  51   0 4.582040874e-12 2.864924299e-10
                 Iteration (num., iy, grad. norm, func.) :  52   0 4.589301761e-12 2.864924259e-10
                 Iteration (num., iy, grad. norm, func.) :  53   0 5.339485232e-12 2.864924255e-10
                 Iteration (num., iy, grad. norm, func.) :  54   0 5.511752329e-12 2.864924248e-10
                 Iteration (num., iy, grad. norm, func.) :  55   0 5.410581025e-12 2.864924233e-10
                 Iteration (num., iy, grad. norm, func.) :  56   0 3.943080143e-12 2.864924206e-10
                 Iteration (num., iy, grad. norm, func.) :  57   0 3.493074738e-12 2.864924191e-10
                 Iteration (num., iy, grad. norm, func.) :  58   0 3.528397901e-12 2.864924182e-10
                 Iteration (num., iy, grad. norm, func.) :  59   0 1.960799706e-12 2.864924176e-10
                 Iteration (num., iy, grad. norm, func.) :  60   0 2.441886155e-12 2.864924174e-10
                 Iteration (num., iy, grad. norm, func.) :  61   0 2.374532068e-12 2.864924170e-10
                 Iteration (num., iy, grad. norm, func.) :  62   0 1.539846288e-12 2.864924165e-10
                 Iteration (num., iy, grad. norm, func.) :  63   0 1.654037809e-12 2.864924163e-10
                 Iteration (num., iy, grad. norm, func.) :  64   0 1.187194957e-12 2.864924161e-10
                 Iteration (num., iy, grad. norm, func.) :  65   0 2.461183662e-12 2.864924158e-10
                 Iteration (num., iy, grad. norm, func.) :  66   0 4.609495090e-13 2.864924155e-10
              Solving for output 0 - done. Time (sec):  2.1466520
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.431922489e-05 6.497521500e-06
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.432062268e-05 6.250717055e-06
                 Iteration (num., iy, grad. norm, func.) :   1   1 1.447473277e-05 8.022139673e-07
                 Iteration (num., iy, grad. norm, func.) :   2   1 1.781373926e-05 3.569835972e-07
                 Iteration (num., iy, grad. norm, func.) :   3   1 5.486407158e-06 1.249456100e-07
                 Iteration (num., iy, grad. norm, func.) :   4   1 9.592921682e-06 1.162559991e-07
                 Iteration (num., iy, grad. norm, func.) :   5   1 5.810703198e-06 7.139269087e-08
                 Iteration (num., iy, grad. norm, func.) :   6   1 1.676464520e-06 3.122631579e-08
                 Iteration (num., iy, grad. norm, func.) :   7   1 1.250439548e-06 3.060168753e-08
                 Iteration (num., iy, grad. norm, func.) :   8   1 7.190277095e-07 2.997510861e-08
                 Iteration (num., iy, grad. norm, func.) :   9   1 3.298304783e-07 2.585302864e-08
                 Iteration (num., iy, grad. norm, func.) :  10   1 1.131350154e-07 1.927223658e-08
                 Iteration (num., iy, grad. norm, func.) :  11   1 6.773522889e-08 1.559978407e-08
                 Iteration (num., iy, grad. norm, func.) :  12   1 2.022425559e-08 1.457641089e-08
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.813611435e-08 1.457612492e-08
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.813611429e-08 1.457612492e-08
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.627997397e-08 1.455608377e-08
                 Iteration (num., iy, grad. norm, func.) :  16   1 1.236953177e-08 1.455081121e-08
                 Iteration (num., iy, grad. norm, func.) :  17   1 2.712358396e-08 1.454380435e-08
                 Iteration (num., iy, grad. norm, func.) :  18   1 5.746887685e-09 1.449786194e-08
                 Iteration (num., iy, grad. norm, func.) :  19   1 4.999029136e-09 1.447934960e-08
                 Iteration (num., iy, grad. norm, func.) :  20   1 5.596398624e-09 1.447383496e-08
                 Iteration (num., iy, grad. norm, func.) :  21   1 1.001964276e-08 1.447323033e-08
                 Iteration (num., iy, grad. norm, func.) :  22   1 4.055311213e-09 1.447109753e-08
                 Iteration (num., iy, grad. norm, func.) :  23   1 3.500278354e-09 1.447072004e-08
                 Iteration (num., iy, grad. norm, func.) :  24   1 3.797276086e-09 1.446927411e-08
                 Iteration (num., iy, grad. norm, func.) :  25   1 3.362438729e-09 1.446679803e-08
                 Iteration (num., iy, grad. norm, func.) :  26   1 2.257794035e-09 1.446518008e-08
                 Iteration (num., iy, grad. norm, func.) :  27   1 1.474682716e-09 1.446466766e-08
                 Iteration (num., iy, grad. norm, func.) :  28   1 1.291801251e-09 1.446462528e-08
                 Iteration (num., iy, grad. norm, func.) :  29   1 1.280225366e-09 1.446454868e-08
                 Iteration (num., iy, grad. norm, func.) :  30   1 1.290729833e-09 1.446426008e-08
                 Iteration (num., iy, grad. norm, func.) :  31   1 1.105759748e-09 1.446397216e-08
                 Iteration (num., iy, grad. norm, func.) :  32   1 1.080263277e-09 1.446390251e-08
                 Iteration (num., iy, grad. norm, func.) :  33   1 7.419467381e-10 1.446385532e-08
                 Iteration (num., iy, grad. norm, func.) :  34   1 9.482963127e-10 1.446382970e-08
                 Iteration (num., iy, grad. norm, func.) :  35   1 6.097762938e-10 1.446377880e-08
                 Iteration (num., iy, grad. norm, func.) :  36   1 8.422762533e-10 1.446373694e-08
                 Iteration (num., iy, grad. norm, func.) :  37   1 4.174569759e-10 1.446367663e-08
                 Iteration (num., iy, grad. norm, func.) :  38   1 4.950108412e-10 1.446363189e-08
                 Iteration (num., iy, grad. norm, func.) :  39   1 2.087152188e-10 1.446360275e-08
                 Iteration (num., iy, grad. norm, func.) :  40   1 2.056686765e-10 1.446359514e-08
                 Iteration (num., iy, grad. norm, func.) :  41   1 2.383612869e-10 1.446359047e-08
                 Iteration (num., iy, grad. norm, func.) :  42   1 2.915449896e-10 1.446358735e-08
                 Iteration (num., iy, grad. norm, func.) :  43   1 2.149396598e-10 1.446358363e-08
                 Iteration (num., iy, grad. norm, func.) :  44   1 2.096948627e-10 1.446357666e-08
                 Iteration (num., iy, grad. norm, func.) :  45   1 1.706834494e-10 1.446357018e-08
                 Iteration (num., iy, grad. norm, func.) :  46   1 1.216256695e-10 1.446356595e-08
                 Iteration (num., iy, grad. norm, func.) :  47   1 1.469374448e-10 1.446356455e-08
                 Iteration (num., iy, grad. norm, func.) :  48   1 8.440355796e-11 1.446356376e-08
                 Iteration (num., iy, grad. norm, func.) :  49   1 7.106318793e-11 1.446356330e-08
                 Iteration (num., iy, grad. norm, func.) :  50   1 7.984321984e-11 1.446356233e-08
                 Iteration (num., iy, grad. norm, func.) :  51   1 7.034230685e-11 1.446356094e-08
                 Iteration (num., iy, grad. norm, func.) :  52   1 4.698252369e-11 1.446356024e-08
                 Iteration (num., iy, grad. norm, func.) :  53   1 6.077638566e-11 1.446356016e-08
                 Iteration (num., iy, grad. norm, func.) :  54   1 7.404934094e-11 1.446356007e-08
                 Iteration (num., iy, grad. norm, func.) :  55   1 3.266337529e-11 1.446355982e-08
                 Iteration (num., iy, grad. norm, func.) :  56   1 4.201151240e-11 1.446355972e-08
                 Iteration (num., iy, grad. norm, func.) :  57   1 2.886621408e-11 1.446355955e-08
                 Iteration (num., iy, grad. norm, func.) :  58   1 3.077501601e-11 1.446355937e-08
                 Iteration (num., iy, grad. norm, func.) :  59   1 1.182083475e-11 1.446355927e-08
                 Iteration (num., iy, grad. norm, func.) :  60   1 1.289100729e-11 1.446355926e-08
                 Iteration (num., iy, grad. norm, func.) :  61   1 1.121472850e-11 1.446355925e-08
                 Iteration (num., iy, grad. norm, func.) :  62   1 1.755529976e-11 1.446355923e-08
                 Iteration (num., iy, grad. norm, func.) :  63   1 9.995399057e-12 1.446355920e-08
                 Iteration (num., iy, grad. norm, func.) :  64   1 1.350331396e-11 1.446355918e-08
                 Iteration (num., iy, grad. norm, func.) :  65   1 8.999488433e-12 1.446355918e-08
                 Iteration (num., iy, grad. norm, func.) :  66   1 6.211355267e-12 1.446355917e-08
                 Iteration (num., iy, grad. norm, func.) :  67   1 7.729934481e-12 1.446355917e-08
                 Iteration (num., iy, grad. norm, func.) :  68   1 6.649950102e-12 1.446355916e-08
                 Iteration (num., iy, grad. norm, func.) :  69   1 7.696764998e-12 1.446355916e-08
                 Iteration (num., iy, grad. norm, func.) :  70   1 3.226810560e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  71   1 4.928725246e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  72   1 3.208713120e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  73   1 4.187869778e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  74   1 2.313904652e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  75   1 2.309390617e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  76   1 1.850313449e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  77   1 1.954888990e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  78   1 1.342160435e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  79   1 2.015479980e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  80   1 9.324053839e-13 1.446355915e-08
              Solving for output 1 - done. Time (sec):  2.5883989
           Solving nonlinear problem (n=1764) - done. Time (sec):  4.7351329
        Solving for degrees of freedom - done. Time (sec):  4.8055439
     Training - done. Time (sec):  4.8276651
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0010209
     
     Prediction time/pt. (sec) :  0.0000020
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009282
     
     Prediction time/pt. (sec) :  0.0000019
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0010998
     
     Prediction time/pt. (sec) :  0.0000022
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0010190
     
     Prediction time/pt. (sec) :  0.0000020
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0010440
     
     Prediction time/pt. (sec) :  0.0000021
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0010209
     
     Prediction time/pt. (sec) :  0.0000020
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0011089
     
     Prediction time/pt. (sec) :  0.0000022
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0010951
     
     Prediction time/pt. (sec) :  0.0000022
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0010939
     
     Prediction time/pt. (sec) :  0.0000022
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0010030
     
     Prediction time/pt. (sec) :  0.0000020
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0011179
     
     Prediction time/pt. (sec) :  0.0000022
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009990
     
     Prediction time/pt. (sec) :  0.0000020
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0010500
     
     Prediction time/pt. (sec) :  0.0000021
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009961
     
     Prediction time/pt. (sec) :  0.0000020
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0036151
     
     Prediction time/pt. (sec) :  0.0000014
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0035200
     
     Prediction time/pt. (sec) :  0.0000014
     
  
.. figure:: rans_crm_wing.png
  :scale: 60 %
  :align: center
