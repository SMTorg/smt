"""
Author: Dr. Mohamed A. Bouhlel <mbouhlel@umich>
        Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

import numpy as np
from scipy import linalg
from smt.utils import compute_rms_error

from smt.problems import Sphere, NdimRobotArm
from smt.sampling_methods import LHS
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS, DesignSpace

try:
    from smt.surrogate_models import IDW, RBF, RMTC, RMTB

    compiled_available = True
except:
    compiled_available = False

try:
    import matplotlib.pyplot as plt

    plot_status = True
except:
    plot_status = False

########### Initialization of the problem, construction of the training and validation points

ndim = 10
ndoe = int(10 * ndim)
# Define the function
fun = Sphere(ndim=ndim)

# Construction of the DOE
sampling = LHS(xlimits=fun.xlimits, criterion="m")
xt = sampling(ndoe)
# Compute the output
yt = fun(xt)
# Compute the gradient
for i in range(ndim):
    yd = fun(xt, kx=i)
    yt = np.concatenate((yt, yd), axis=1)

# Construction of the validation points
ntest = 500
sampling = LHS(xlimits=fun.xlimits)
xtest = sampling(ntest)
ytest = fun(xtest)
ydtest = np.zeros((ntest, ndim))
for i in range(ndim):
    ydtest[:, i] = fun(xtest, kx=i).T


########### The LS model

# Initialization of the model
t = LS(print_prediction=False)
# Add the DOE
t.set_training_values(xt, yt[:, 0])

# Train the model
t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print("LS,  err: " + str(compute_rms_error(t, xtest, ytest)))

if plot_status:
    k, l = 0, 0
    f, axarr = plt.subplots(4, 3)
    axarr[k, l].plot(ytest, ytest, "-.")
    axarr[k, l].plot(ytest, y, ".")
    l += 1
    axarr[3, 2].arrow(0.3, 0.3, 0.2, 0)
    axarr[3, 2].arrow(0.3, 0.3, 0.0, 0.4)
    axarr[3, 2].text(0.25, 0.4, r"$\hat{y}$")
    axarr[3, 2].text(0.35, 0.15, r"$y_{true}$")
    axarr[3, 2].axis("off")
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp(axarr[3, 2].get_xticklabels(), visible=False)
    plt.setp(axarr[3, 2].get_yticklabels(), visible=False)
    plt.suptitle(
        "Validation of the LS model (from left to right then from top to bottom): validation of the prediction model and the i-th prediction of the derivative---i=1:10"
    )


# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest, ndim))
for i in range(ndim):
    yd_prediction[:, i] = t.predict_derivatives(xtest, kx=i).T
    print(
        "LS, err of the "
        + str(i)
        + "-th derivative: "
        + str(compute_rms_error(t, xtest, ydtest[:, i], kx=i))
    )

    if plot_status:
        axarr[k, l].plot(ydtest[:, i], ydtest[:, i], "-.")
        axarr[k, l].plot(ydtest[:, i], yd_prediction[:, i], ".")
        if l == 2:
            l = 0
            k += 1
        else:
            l += 1

if plot_status:
    plt.show()

########### The QP model

t = QP(print_prediction=False)
t.set_training_values(xt, yt[:, 0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print("QP,  err: " + str(compute_rms_error(t, xtest, ytest)))
if plot_status:
    k, l = 0, 0
    f, axarr = plt.subplots(4, 3)
    axarr[k, l].plot(ytest, ytest, "-.")
    axarr[k, l].plot(ytest, y, ".")
    l += 1
    axarr[3, 2].arrow(0.3, 0.3, 0.2, 0)
    axarr[3, 2].arrow(0.3, 0.3, 0.0, 0.4)
    axarr[3, 2].text(0.25, 0.4, r"$\hat{y}$")
    axarr[3, 2].text(0.35, 0.15, r"$y_{true}$")
    axarr[3, 2].axis("off")
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp(axarr[3, 2].get_xticklabels(), visible=False)
    plt.setp(axarr[3, 2].get_yticklabels(), visible=False)
    plt.suptitle(
        "Validation of the QP model (from left to right then from top to bottom): validation of the prediction model and the i-th prediction of the derivative---i=1:10"
    )


# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest, ndim))
for i in range(ndim):
    yd_prediction[:, i] = t.predict_derivatives(xtest, kx=i).T
    print(
        "QP, err of the "
        + str(i)
        + "-th derivative: "
        + str(compute_rms_error(t, xtest, ydtest[:, i], kx=i))
    )

    if plot_status:
        axarr[k, l].plot(ydtest[:, i], ydtest[:, i], "-.")
        axarr[k, l].plot(ydtest[:, i], yd_prediction[:, i], ".")
        if l == 2:
            l = 0
            k += 1
        else:
            l += 1

if plot_status:
    plt.show()

########### The Kriging model

# The variable 'theta0' is a list of length ndim.
t = KRG(theta0=[1e-2] * ndim, print_prediction=False)
t.set_training_values(xt, yt[:, 0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print("Kriging,  err: " + str(compute_rms_error(t, xtest, ytest)))
if plot_status:
    k, l = 0, 0
    f, axarr = plt.subplots(4, 3)
    axarr[k, l].plot(ytest, ytest, "-.")
    axarr[k, l].plot(ytest, y, ".")
    l += 1
    axarr[3, 2].arrow(0.3, 0.3, 0.2, 0)
    axarr[3, 2].arrow(0.3, 0.3, 0.0, 0.4)
    axarr[3, 2].text(0.25, 0.4, r"$\hat{y}$")
    axarr[3, 2].text(0.35, 0.15, r"$y_{true}$")
    axarr[3, 2].axis("off")
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp(axarr[3, 2].get_xticklabels(), visible=False)
    plt.setp(axarr[3, 2].get_yticklabels(), visible=False)
    plt.suptitle(
        "Validation of the Kriging model (from left to right then from top to bottom): validation of the prediction model and the i-th prediction of the derivative---i=1:10"
    )


# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest, ndim))
for i in range(ndim):
    yd_prediction[:, i] = t.predict_derivatives(xtest, kx=i).T
    print(
        "Kriging, err of the "
        + str(i)
        + "-th derivative: "
        + str(compute_rms_error(t, xtest, ydtest[:, i], kx=i))
    )

    if plot_status:
        axarr[k, l].plot(ydtest[:, i], ydtest[:, i], "-.")
        axarr[k, l].plot(ydtest[:, i], yd_prediction[:, i], ".")
        if l == 2:
            l = 0
            k += 1
        else:
            l += 1

if plot_status:
    plt.show()

########### The KPLS model

# The variables 'name' must be equal to 'KPLS'. 'n_comp' and 'theta0' must be
# an integer in [1, ndim[ and a list of length n_comp, respectively. Here is an
# an example using 2 principal components.

t = KPLS(n_comp=2, theta0=[1e-2, 1e-2], print_prediction=False)
t.set_training_values(xt, yt[:, 0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print("KPLS,  err: " + str(compute_rms_error(t, xtest, ytest)))
if plot_status:
    k, l = 0, 0
    f, axarr = plt.subplots(4, 3)
    axarr[k, l].plot(ytest, ytest, "-.")
    axarr[k, l].plot(ytest, y, ".")
    l += 1
    axarr[3, 2].arrow(0.3, 0.3, 0.2, 0)
    axarr[3, 2].arrow(0.3, 0.3, 0.0, 0.4)
    axarr[3, 2].text(0.25, 0.4, r"$\hat{y}$")
    axarr[3, 2].text(0.35, 0.15, r"$y_{true}$")
    axarr[3, 2].axis("off")
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp(axarr[3, 2].get_xticklabels(), visible=False)
    plt.setp(axarr[3, 2].get_yticklabels(), visible=False)
    plt.suptitle(
        "Validation of the KPLS model (from left to right then from top to bottom): validation of the prediction model and the i-th prediction of the derivative---i=1:10"
    )

# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest, ndim))
for i in range(ndim):
    yd_prediction[:, i] = t.predict_derivatives(xtest, kx=i).T
    print(
        "KPLS, err of the "
        + str(i)
        + "-th derivative: "
        + str(compute_rms_error(t, xtest, ydtest[:, i], kx=i))
    )

    if plot_status:
        axarr[k, l].plot(ydtest[:, i], ydtest[:, i], "-.")
        axarr[k, l].plot(ydtest[:, i], yd_prediction[:, i], ".")
        if l == 2:
            l = 0
            k += 1
        else:
            l += 1

if plot_status:
    plt.show()

# KPLS + absolute exponential correlation kernel
# The variables 'name' must be equal to 'KPLS'. 'n_comp' and 'theta0' must be
# an integer in [1,ndim[ and a list of length n_comp, respectively. Here is an
# an example using 2 principal components.

t = KPLS(n_comp=2, theta0=[1e-2, 1e-2], print_prediction=False, corr="abs_exp")
t.set_training_values(xt, yt[:, 0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print("KPLS + abs exp,  err: " + str(compute_rms_error(t, xtest, ytest)))


########### The KPLSK model

# 'n_comp' and 'theta0' must be an integer in [1, ndim[ and a list of length n_comp, respectively.

t = KPLSK(n_comp=2, theta0=[1e-2, 1e-2], print_prediction=False)
t.set_training_values(xt, yt[:, 0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print("KPLSK,  err: " + str(compute_rms_error(t, xtest, ytest)))
if plot_status:
    k, l = 0, 0
    f, axarr = plt.subplots(4, 3)
    axarr[k, l].plot(ytest, ytest, "-.")
    axarr[k, l].plot(ytest, y, ".")
    l += 1
    axarr[3, 2].arrow(0.3, 0.3, 0.2, 0)
    axarr[3, 2].arrow(0.3, 0.3, 0.0, 0.4)
    axarr[3, 2].text(0.25, 0.4, r"$\hat{y}$")
    axarr[3, 2].text(0.35, 0.15, r"$y_{true}$")
    axarr[3, 2].axis("off")
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp(axarr[3, 2].get_xticklabels(), visible=False)
    plt.setp(axarr[3, 2].get_yticklabels(), visible=False)
    plt.suptitle(
        "Validation of the KPLSK model (from left to right then from top to bottom): validation of the prediction model and the i-th prediction of the derivative---i=1:10"
    )


# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest, ndim))
for i in range(ndim):
    yd_prediction[:, i] = t.predict_derivatives(xtest, kx=i).T
    print(
        "KPLSK, err of the "
        + str(i)
        + "-th derivative: "
        + str(compute_rms_error(t, xtest, ydtest[:, i], kx=i))
    )

    if plot_status:
        axarr[k, l].plot(ydtest[:, i], ydtest[:, i], "-.")
        axarr[k, l].plot(ydtest[:, i], yd_prediction[:, i], ".")
        if l == 2:
            l = 0
            k += 1
        else:
            l += 1

if plot_status:
    plt.show()

########### The GEKPLS model using 1 approximating points

# 'n_comp' and 'theta0' must be an integer in [1,ndim[ and a list of length n_comp, respectively.
design_space = DesignSpace(fun.xlimits)
t = GEKPLS(
    n_comp=1,
    theta0=[1e-2],
    design_space=design_space,
    delta_x=1e-2,
    extra_points=1,
    print_prediction=False,
)
t.set_training_values(xt, yt[:, 0])
# Add the gradient information
for i in range(ndim):
    t.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print("GEKPLS1,  err: " + str(compute_rms_error(t, xtest, ytest)))
if plot_status:
    k, l = 0, 0
    f, axarr = plt.subplots(4, 3)
    axarr[k, l].plot(ytest, ytest, "-.")
    axarr[k, l].plot(ytest, y, ".")
    l += 1
    axarr[3, 2].arrow(0.3, 0.3, 0.2, 0)
    axarr[3, 2].arrow(0.3, 0.3, 0.0, 0.4)
    axarr[3, 2].text(0.25, 0.4, r"$\hat{y}$")
    axarr[3, 2].text(0.35, 0.15, r"$y_{true}$")
    axarr[3, 2].axis("off")
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp(axarr[3, 2].get_xticklabels(), visible=False)
    plt.setp(axarr[3, 2].get_yticklabels(), visible=False)
    plt.suptitle(
        "Validation of the GEKPLS1 model (from left to right then from top to bottom): validation of the prediction model and the i-th prediction of the derivative---i=1:10"
    )


# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest, ndim))
for i in range(ndim):
    yd_prediction[:, i] = t.predict_derivatives(xtest, kx=i).T
    print(
        "GEKPLS1, err of the "
        + str(i)
        + "-th derivative: "
        + str(compute_rms_error(t, xtest, ydtest[:, i], kx=i))
    )

    if plot_status:
        axarr[k, l].plot(ydtest[:, i], ydtest[:, i], "-.")
        axarr[k, l].plot(ydtest[:, i], yd_prediction[:, i], ".")
        if l == 2:
            l = 0
            k += 1
        else:
            l += 1

if plot_status:
    plt.show()

########### The GEKPLS model using 2 approximating points

# 'n_comp' and 'theta0' must be an integer in [1,ndim[ and a list of length n_comp, respectively.

t = GEKPLS(
    n_comp=1,
    theta0=[1e-2],
    xlimits=fun.xlimits,
    delta_x=1e-4,
    extra_points=2,
    print_prediction=False,
)
t.set_training_values(xt, yt[:, 0])
# Add the gradient information
for i in range(ndim):
    t.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print("GEKPLS2,  err: " + str(compute_rms_error(t, xtest, ytest)))
if plot_status:
    k, l = 0, 0
    f, axarr = plt.subplots(4, 3)
    axarr[k, l].plot(ytest, ytest, "-.")
    axarr[k, l].plot(ytest, y, ".")
    l += 1
    axarr[3, 2].arrow(0.3, 0.3, 0.2, 0)
    axarr[3, 2].arrow(0.3, 0.3, 0.0, 0.4)
    axarr[3, 2].text(0.25, 0.4, r"$\hat{y}$")
    axarr[3, 2].text(0.35, 0.15, r"$y_{true}$")
    axarr[3, 2].axis("off")
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp(axarr[3, 2].get_xticklabels(), visible=False)
    plt.setp(axarr[3, 2].get_yticklabels(), visible=False)
    plt.suptitle(
        "Validation of the GEKPLS2 model (from left to right then from top to bottom): validation of the prediction model and the i-th prediction of the derivative---i=1:10"
    )


# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest, ndim))
for i in range(ndim):
    yd_prediction[:, i] = t.predict_derivatives(xtest, kx=i).T
    print(
        "GEKPLS2, err of the "
        + str(i)
        + "-th derivative: "
        + str(compute_rms_error(t, xtest, ydtest[:, i], kx=i))
    )

    if plot_status:
        axarr[k, l].plot(ydtest[:, i], ydtest[:, i], "-.")
        axarr[k, l].plot(ydtest[:, i], yd_prediction[:, i], ".")
        if l == 2:
            l = 0
            k += 1
        else:
            l += 1

if plot_status:
    plt.show()
if compiled_available:
    ########### The IDW model

    t = IDW(print_prediction=False)
    t.set_training_values(xt, yt[:, 0])

    t.train()

    # Prediction of the validation points
    y = t.predict_values(xtest)
    print("IDW,  err: " + str(compute_rms_error(t, xtest, ytest)))
    if plot_status:
        plt.figure()
        plt.plot(ytest, ytest, "-.")
        plt.plot(ytest, y, ".")
        plt.xlabel(r"$y_{true}$")
        plt.ylabel(r"$\hat{y}$")
        plt.title("Validation of the IDW model")
        plt.show()

    ########### The RBF model

    t = RBF(print_prediction=False, poly_degree=0)
    t.set_training_values(xt, yt[:, 0])

    t.train()

    # Prediction of the validation points
    y = t.predict_values(xtest)
    print("RBF,  err: " + str(compute_rms_error(t, xtest, ytest)))
    if plot_status:
        k, l = 0, 0
        f, axarr = plt.subplots(4, 3)
        axarr[k, l].plot(ytest, ytest, "-.")
        axarr[k, l].plot(ytest, y, ".")
        l += 1
        axarr[3, 2].arrow(0.3, 0.3, 0.2, 0)
        axarr[3, 2].arrow(0.3, 0.3, 0.0, 0.4)
        axarr[3, 2].text(0.25, 0.4, r"$\hat{y}$")
        axarr[3, 2].text(0.35, 0.15, r"$y_{true}$")
        axarr[3, 2].axis("off")
        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        plt.setp(axarr[3, 2].get_xticklabels(), visible=False)
        plt.setp(axarr[3, 2].get_yticklabels(), visible=False)
        plt.suptitle(
            "Validation of the RBF model (from left to right then from top to bottom): validation of the prediction model and the i-th prediction of the derivative---i=1:10"
        )

    # Prediction of the derivatives with regards to each direction space
    yd_prediction = np.zeros((ntest, ndim))
    for i in range(ndim):
        yd_prediction[:, i] = t.predict_derivatives(xtest, kx=i).T
        print(
            "RBF, err of the "
            + str(i)
            + "-th derivative: "
            + str(compute_rms_error(t, xtest, ydtest[:, i], kx=i))
        )

        if plot_status:
            axarr[k, l].plot(ydtest[:, i], ydtest[:, i], "-.")
            axarr[k, l].plot(ydtest[:, i], yd_prediction[:, i], ".")
            if l == 2:
                l = 0
                k += 1
            else:
                l += 1

    if plot_status:
        plt.show()

    ########### The RMTB and RMTC models are suitable for low-dimensional problems

    # Initialization of the problem
    ndim = 3
    ndoe = int(250 * ndim)
    # Define the function
    fun = NdimRobotArm(ndim=ndim)

    # Construction of the DOE
    sampling = LHS(xlimits=fun.xlimits)
    xt = sampling(ndoe)

    # Compute the output
    yt = fun(xt)
    # Compute the gradient
    for i in range(ndim):
        yd = fun(xt, kx=i)
        yt = np.concatenate((yt, yd), axis=1)

    # Construction of the validation points
    ntest = 500
    sampling = LHS(xlimits=fun.xlimits)
    xtest = sampling(ntest)
    ytest = fun(xtest)

    ########### The RMTB model

    t = RMTB(
        xlimits=fun.xlimits,
        min_energy=True,
        nonlinear_maxiter=20,
        print_prediction=False,
    )
    t.set_training_values(xt, yt[:, 0])
    # Add the gradient information
    for i in range(ndim):
        t.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)
    t.train()

    # Prediction of the validation points
    y = t.predict_values(xtest)
    print("RMTB,  err: " + str(compute_rms_error(t, xtest, ytest)))
    if plot_status:
        k, l = 0, 0
        f, axarr = plt.subplots(3, 2)
        axarr[k, l].plot(ytest, ytest, "-.")
        axarr[k, l].plot(ytest, y, ".")
        l += 1
        axarr[2, 0].arrow(0.3, 0.3, 0.2, 0)
        axarr[2, 0].arrow(0.3, 0.3, 0.0, 0.4)
        axarr[2, 0].text(0.25, 0.4, r"$\hat{y}$")
        axarr[2, 0].text(0.35, 0.15, r"$y_{true}$")
        axarr[2, 0].axis("off")
        axarr[2, 1].set_visible(False)
        axarr[2, 1].axis("off")

        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        plt.setp(axarr[2, 0].get_xticklabels(), visible=False)
        plt.setp(axarr[2, 0].get_yticklabels(), visible=False)
        plt.suptitle(
            "Validation of the RMTB model (from left to right then from top to bottom): validation of the prediction model and the i-th prediction of the derivative---i=1:3"
        )

    # Prediction of the derivatives with regards to each direction space
    yd_prediction = np.zeros((ntest, ndim))
    for i in range(ndim):
        yd_prediction[:, i] = t.predict_derivatives(xtest, kx=i).T
        print(
            "RMTB, err of the "
            + str(i)
            + "-th derivative: "
            + str(compute_rms_error(t, xtest, ydtest[:, i], kx=i))
        )

        if plot_status:
            axarr[k, l].plot(ydtest[:, i], ydtest[:, i], "-.")
            axarr[k, l].plot(ydtest[:, i], yd_prediction[:, i], ".")
            if l == 1:
                l = 0
                k += 1
            else:
                l += 1

    if plot_status:
        plt.show()

    ########### The RMTC model

    t = RMTC(
        xlimits=fun.xlimits,
        min_energy=True,
        nonlinear_maxiter=20,
        print_prediction=False,
    )
    t.set_training_values(xt, yt[:, 0])
    # Add the gradient information
    for i in range(ndim):
        t.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)

    t.train()

    # Prediction of the validation points
    y = t.predict_values(xtest)
    print("RMTC,  err: " + str(compute_rms_error(t, xtest, ytest)))
    if plot_status:
        k, l = 0, 0
        f, axarr = plt.subplots(3, 2)
        axarr[k, l].plot(ytest, ytest, "-.")
        axarr[k, l].plot(ytest, y, ".")
        l += 1
        axarr[2, 0].arrow(0.3, 0.3, 0.2, 0)
        axarr[2, 0].arrow(0.3, 0.3, 0.0, 0.4)
        axarr[2, 0].text(0.25, 0.4, r"$\hat{y}$")
        axarr[2, 0].text(0.35, 0.15, r"$y_{true}$")
        axarr[2, 0].axis("off")
        axarr[2, 1].set_visible(False)
        axarr[2, 1].axis("off")

        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        plt.setp(axarr[2, 0].get_xticklabels(), visible=False)
        plt.setp(axarr[2, 0].get_yticklabels(), visible=False)
        plt.suptitle(
            "Validation of the RMTC model (from left to right then from top to bottom): validation of the prediction model and the i-th prediction of the derivative---i=1:3"
        )

    # Prediction of the derivatives with regards to each direction space
    yd_prediction = np.zeros((ntest, ndim))
    for i in range(ndim):
        yd_prediction[:, i] = t.predict_derivatives(xtest, kx=i).T
        print(
            "RMTC, err of the "
            + str(i)
            + "-th derivative: "
            + str(compute_rms_error(t, xtest, ydtest[:, i], kx=i))
        )

        if plot_status:
            axarr[k, l].plot(ydtest[:, i], ydtest[:, i], "-.")
            axarr[k, l].plot(ydtest[:, i], yd_prediction[:, i], ".")
            if l == 1:
                l = 0
                k += 1
            else:
                l += 1

    if plot_status:
        plt.show()
