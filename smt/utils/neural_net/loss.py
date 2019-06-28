"""
G R A D I E N T - E N H A N C E D   N E U R A L   N E T W O R K S  (G E N N)

Author: Steven H. Berguin <steven.berguin@gtri.gatech.edu>

This package is distributed under New BSD license.
"""

import numpy as np

tensor = np.ndarray

EPS = np.finfo(float).eps  # small number to avoid division by zero


def compute_regularization(w, lambd=0.0):
    """
    Compute L2 norm penalty

    :param: w: the weight parameters of each layer of the neural net
    :param: lambd: float, regularization coefficient
    :return: penalty: np.ndarray of shape (1,)
    """
    lambd = max(0.0, lambd)  # ensure 0 < lambda
    penalty = 0.0
    for theta in w:
        penalty += np.squeeze(0.5 * lambd * np.sum(np.square(theta)))
    return penalty


def compute_gradient_enhancement(dy_true, dy_pred, gamma=0.0):
    """
    Compute gradient enhancement term (apply LSE to partials)

    :param: dy_pred: np ndarray of shape (n_y, n_x, m) -- predicted partials: AL' = d(AL)/dX
                                                               where n_y = # outputs, n_x = # inputs, m = # examples

    :param: dy_true: np ndarray of shape (n_y, n_x, m) -- true partials: Y' = d(Y)/dX
                                                          where n_y = # outputs, n_x = # inputs, m = # examples

    :return: loss: np.ndarray of shape (1,)
    """
    n_y, n_x, m = dy_pred.shape  # number of outputs, inputs, training examples
    loss = 0.0
    gamma = min(max(0.0, gamma), 1.0)  # ensure 0 < gamma < 1
    for k in range(0, n_y):
        for j in range(0, n_x):
            dy_j_pred = dy_pred[k, j, :].reshape(1, m)
            dy_j_true = dy_true[k, j, :].reshape(1, m)
            loss += np.squeeze(
                0.5 * gamma * np.dot((dy_j_pred - dy_j_true), (dy_j_pred - dy_j_true).T)
            )

    return loss


def lse(y_true, y_pred, lambd=0.0, w=None, dy_true=None, dy_pred=None, gamma=0.0):
    """
    Compute least squares estimator loss for regression

    :param: y_pred: np ndarray of shape (n_y, m) -- output of the forward propagation L_model_forward()
                                                    where n_y = no. outputs, m = no. examples

    :param: y_true: np ndarray of shape (n_y, m) -- true labels (classification) or values (regression)
                                               where n_y = no. outputs, m = no. examples
    :return: loss: np.ndarray of shape (1,)
    """
    n_y, m = y_true.shape  # number of outputs, training examples
    cost = 0.0
    for k in range(0, n_y):
        cost += np.squeeze(
            0.5 * np.dot((y_pred[k, :] - y_true[k, :]), (y_pred[k, :] - y_true[k, :]).T)
        )

    if w is not None:
        cost += compute_regularization(w, lambd)

    if dy_true is not None and dy_pred is not None:
        cost += compute_gradient_enhancement(dy_true, dy_pred, gamma)

    return 1.0 / m * cost


if __name__ == "__main__":  # pragma: no cover

    # Check that LSE computes correctly
    w = [np.array(1.0), np.array(2.0)]
    f = lambda x: w[0] * x + w[1] * x ** 2
    dfdx = lambda x: w[0] + 2 * w[1] * x
    m = 100
    lb = -5.0
    ub = 5.0
    x = np.linspace(lb, ub, m)
    y_true = f(x).reshape(1, m)
    y_pred = f(x).reshape(1, m) + 1.0
    dy_true = dfdx(x).reshape(1, 1, m)
    dy_pred = dfdx(x).reshape(1, 1, m) + 1.0
    loss = lse(
        y_true=y_true,
        y_pred=y_pred,
        dy_true=dy_true,
        dy_pred=dy_pred,
        w=w,
        lambd=1.0,
        gamma=1.0,
    )
    assert loss == 1.025
