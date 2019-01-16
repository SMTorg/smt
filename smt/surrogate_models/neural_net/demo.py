"""
G R A D I E N T - E N H A N C E D   N E U R A L   N E T W O R K S  (G E N N)

Author: Steven H. Berguin <steven.berguin@gtri.gatech.edu>

This package is distributed under New BSD license.
"""

from smt.surrogate_models.neural_net.model import Model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyDOE2 import fullfact


def get_practice_data(random=False):
    """
    Return practice data for two-dimensional Rastrigin function

    :param: random -- boolean, True = random sampling, False = full-factorial sampling
    :return: (X, Y, J) -- np arrays of shapes (n_x, m), (n_y, m), (n_y, n_x, m) where n_x = 2 and n_y = 1 and m = 15^2
    """
    # Response (N-dimensional Rastrigin)
    f = lambda x: np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10, axis=1)
    df = lambda x, j: 2 * x[:, j] + 20 * np.pi * np.sin(2 * np.pi * x[:, j])

    # Domain
    lb = -1.  # minimum bound (same for all dimensions)
    ub = 1.5  # maximum bound (same for all dimensions)

    # Design of experiment (full factorial)
    n_x = 2  # number of dimensions
    n_y = 1  # number of responses
    L = 12  # number of levels per dimension
    m = L ** n_x  # number of training examples that will be generated

    if random:
        doe = np.random.rand(m, n_x)
    else:
        levels = [L] * n_x
        doe = fullfact(levels)
        doe = (doe - 0.) / (L - 1.)  # values normalized such that 0 < doe < 1

    assert (doe.shape == (m, n_x))

    # Apply bounds
    X = lb + (ub - lb) * doe

    # Evaluate response
    Y = f(X).reshape((m, 1))

    # Evaluate partials
    J = np.zeros((m, n_x, n_y))
    for j in range(0, n_x):
        J[:, j, :] = df(X, j).reshape((m, 1))

    return X.T, Y.T, J.T


def contour_plot(model, title='GENN'):
    """Make contour plots of 2D Rastrigin function and compare to Neural Net prediction"""

    X_train, _, _ = model.training_data

    # Domain
    lb = -1.
    ub = 1.5
    m = 100
    x1 = np.linspace(lb, ub, m)
    x2 = np.linspace(lb, ub, m)
    X1, X2 = np.meshgrid(x1, x2)

    # True response
    pi = np.pi
    Y_true = np.power(X1, 2) - 10 * np.cos(2 * pi * X1) + 10 + np.power(X2, 2) - 10 * np.cos(2 * pi * X2) + 10

    # Predicted response
    Y_pred = np.zeros((m, m))
    for i in range(0, m):
        for j in range(0, m):
            Y_pred[i, j] = model.evaluate(np.array([X1[i, j], X2[i, j]]).reshape(2, 1))

    # Prepare to plot
    fig = plt.figure(figsize=(6, 3))
    spec = gridspec.GridSpec(ncols=2, nrows=1, wspace=0)

    # Plot Truth model
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.contour(X1, X2, Y_true, 20, cmap='RdGy')
    anno_opts = dict(xy=(0.5, 1.075), xycoords='axes fraction', va='center', ha='center')
    ax1.annotate('True', **anno_opts)
    anno_opts = dict(xy=(-0.075, 0.5), xycoords='axes fraction', va='center', ha='center')
    ax1.annotate('X2', **anno_opts)
    anno_opts = dict(xy=(0.5, -0.05), xycoords='axes fraction', va='center', ha='center')
    ax1.annotate('X1', **anno_opts)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.scatter(X_train[0, :], X_train[1, :], s=5)
    ax1.set_xlim(lb, ub)
    ax1.set_ylim(lb, ub)

    # Plot prediction with gradient enhancement
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.contour(X1, X2, Y_pred, 20, cmap='RdGy')
    anno_opts = dict(xy=(0.5, 1.075), xycoords='axes fraction', va='center', ha='center')
    ax2.annotate(title, **anno_opts)
    anno_opts = dict(xy=(0.5, -0.05), xycoords='axes fraction', va='center', ha='center')
    ax2.annotate('X1', **anno_opts)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.show()


def main(alpha=0.1, beta1=0.9, beta2=0.99, lambd=0.1, gamma=1.0, deep=3, wide=6, batches=32, iterations=30, epochs=100):
    """
    Predict Rastrigin function using neural net and compare against truth model. Provided with proper training data,
    the only hyperparameters the user needs to tune are:

    :param alpha = learning rate
    :param beta1 = adam optimizer parameter
    :param beta2 = adam optimizer parameter
    :param lambd = regularization coefficient
    :param gamma = gradient enhancement coefficient
    :param deep = neural net depth
    :param wide = neural net width

    This restricted list is intentional. The goal was to provide a simple interface for common regression tasks
    with the bare necessary tuning parameters. More advanced prediction tasks should consider tensorflow or other
    deep learning frameworks. Hopefully, the simplicity of this interface will address a common use case in aerospace
    engineering, namely: predicting smooth functions using computational design of experiments.
    """
    if gamma > 0.:
        title = 'GENN'
    else:
        title = 'NN'

    # Practice data
    X_train, Y_train, J_train = get_practice_data(random=False)
    X_test, Y_test, J_test = get_practice_data(random=True)

    # Training
    model = Model.initialize(n_x=X_train.shape[0],
                             n_y=Y_train.shape[0],
                             deep=deep,
                             wide=wide)
    model.train(X=X_train,
                Y=Y_train,
                J=J_train,
                alpha=alpha,
                beta1=beta1,
                beta2=beta2,
                lambd=lambd,
                gamma=gamma,
                mini_batch_size=batches,
                num_iterations=iterations,
                num_epochs=epochs,
                silent=False)
    model.plot_training_history()
    model.print_training_history()
    model.goodness_of_fit(X_test, Y_test)  # model.goodness_of_fit(X_test, Y_test, J_test, partial=1)
    model.print_parameters()

    # Contour plot
    contour_plot(model, title=title)


if __name__ == "__main__":
    main(alpha=0.1,
         beta1=0.9,
         beta2=0.99,
         lambd=0.1,
         gamma=1.0,
         deep=3,
         wide=6,
         batches=32,
         iterations=30,
         epochs=50)

