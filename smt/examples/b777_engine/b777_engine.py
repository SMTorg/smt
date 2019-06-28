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
