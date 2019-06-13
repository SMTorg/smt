"""
G R A D I E N T - E N H A N C E D   N E U R A L   N E T W O R K S  (G E N N)

Author: Steven H. Berguin <steven.berguin@gtri.gatech.edu>

This package is distributed under New BSD license.
"""

import numpy as np

EPS = np.finfo(float).eps  # small number to avoid division by zero


def initialize_back_prop(AL, Y, AL_prime, Y_prime):
    """
    Initialize backward propagation

    Arguments:
    :param AL -- output of the forward propagation L_model_forward()... i.e. neural net predictions   (if regression)
                                                                 i.e. neural net probabilities (if classification)
          >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    :param Y -- true "label" (classification) or "value" (regression)
         >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    :param AL_prime -- the derivative of the last layer's activation output(s) w.r.t. the inputs x: AL' = d(AL)/dX
                >> a numpy array of size (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    :param Y_prime -- the true derivative of the output(s) w.r.t. the inputs x: Y' = d(Y)/dX
               >> a numpy array of shape (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    Returns:
    :return dAL -- gradient of the loss function w.r.t. last layer activations: d(L)/dAL
           >> a numpy array of shape (n_y, m)

    :return dAL_prime -- gradient of the loss function w.r.t. last layer activations derivatives: d(L)/dAL' where AL' = d(AL)/dX
                 >> a numpy array of shape (n_y, n_x, m)
    """
    n_y, _ = AL.shape  # number layers, number examples
    Y = Y.reshape(AL.shape)
    dAL = AL - Y  # derivative of loss function w.r.t. to activations: dAL = d(L)/dAL
    dAL_prime = (
        AL_prime - Y_prime
    )  # derivative of loss function w.r.t. to partials: dAL_prime = d(L)/d(AL_prime)

    return dAL, dAL_prime


def linear_activation_backward(dA, dA_prime, cache, J_cache, lambd, gamma):
    """
    Implement backward propagation for one LINEAR->ACTIVATION layer for the regression least squares estimation

    Arguments:
    :param dA -- post-activation gradient w.r.t. A for current layer l, dA = d(L)/dA where L is the loss function
            >> a numpy array of shape (n_1, m) where n_l = no. nodes in current layer, m = no. of examples

    :param dA_prime -- post-activation gradient w.r.t. A' for current layer l, dA' = d(L)/dA' where L is the loss function
                                                                                        and A' = d(AL) / dX
            >> a numpy array of shape (n_l, n_x, m) where n_l = no. nodes in current layer
                                                          n_x = no. of inputs (X1, X2, ...)
                                                          m = no. of examples

    :param cache -- tuple of values stored in linear_activation_forward()
              >> a tuple containing (A_prev, Z, W, b, activation)
                      where
                            A_prev -- activations from previous layer
                                      >> a numpy array of shape (n_prev, m) where n_prev is the no. nodes in layer L-1
                            Z -- input to activation functions for current layer
                                      >> a numpy array of shape (n, m) where n is the no. nodes in layer L
                            W -- weight parameters for current layer
                                 >> a numpy array of shape (n, n_prev)
                            b -- bias parameters for current layer
                                 >> a numpy array of shape (n, 1)
                            activation -- activation function to use

    :param J_cache -- list of caches containing every cache of L_grads_forward() where J stands for Jacobian
              >> a list containing [..., (j, Z_prime_j, A_prime_j, G_prime, G_prime_prime), ...]
                                          ------------------ input j --------------------
                      where
                            j -- input variable associated with current cache
                                      >> an integer representing the associated input variables (X1, X2, ..., Xj, ...)
                            Z_prime_j -- derivative of Z w.r.t. X_j: Z'_j = d(Z_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l
                            A_prime_j -- derivative of the activation w.r.t. X_j: A'_j = d(A_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l

    :param lambd: float, regularization parameter
    :param gamma: float, gradient-enhancement parameter

    :return dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    :return dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    :return db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    # Extract information from current layer cache (avoids recomputing what was previously computed)
    A_prev, Z, W, b, activation = cache

    # Some dimensions that will be useful
    m = A_prev.shape[1]  # number of examples
    n = len(J_cache)  # number of inputs

    # 1st derivative of activation function A = G(Z)
    G_prime = activation.first_derivative(Z)

    # Compute the contribution due to the 0th order terms (where regularization only affects dW)
    dW = (
        1.0 / m * np.dot(G_prime * dA, A_prev.T) + lambd / m * W
    )  # dW = d(J)/dW where J is the cost function
    db = 1.0 / m * np.sum(G_prime * dA, axis=1, keepdims=True)  # db = d(J)/db
    dA_prev = np.dot(
        W.T, G_prime * dA
    )  # dA_prev = d(L)/dA_prev where A_prev = previous layer activation

    # Initialize dA_prime_prev = d(J)/dA_prime_prev
    dA_prime_prev = np.zeros((W.shape[1], n, m))

    # Gradient enhancement
    if gamma != 0:

        # 2nd derivative of activation function A = G(Z)
        G_prime_prime = activation.second_derivative(Z)

        # Loop over partials, d()/dX_j
        for j_cache in J_cache:
            # Extract information from current layer cache associated with derivative of A w.r.t. j^th input
            j, Z_prime_j, A_prime_j_prev = j_cache

            # Extract partials of A w.r.t. to j^th input, i.e. A_prime_j = d(A)/dX_j
            dA_prime_j = dA_prime[:, j, :].reshape(Z_prime_j.shape)

            # Compute contribution to cost function gradient, db = d(J)/db, dW = d(J)/dW, d(L)/dA, d(L)/dA'
            dW += (
                gamma
                / m
                * (
                    np.dot(dA_prime_j * G_prime_prime * Z_prime_j, A_prev.T)
                    + np.dot(dA_prime_j * G_prime, A_prime_j_prev.T)
                )
            )
            db += (
                gamma
                / m
                * np.sum(dA_prime_j * G_prime_prime * Z_prime_j, axis=1, keepdims=True)
            )
            dA_prev += gamma * np.dot(W.T, dA_prime_j * G_prime_prime * Z_prime_j)
            dA_prime_prev[:, j, :] = gamma * np.dot(W.T, dA_prime_j * G_prime)

    return dA_prev, dW, db, dA_prime_prev


def L_model_backward(AL, Y, AL_prime, Y_prime, caches, J_caches, lambd, gamma):
    """
    Implement backward propagation

    Arguments:
    :param AL -- output of the forward propagation L_model_forward()... i.e. neural net predictions   (if regression)
                                                                 i.e. neural net probabilities (if classification)
          >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    :param Y -- true "label" (classification) or "value" (regression)
         >> a numpy array of shape (n_y, m) where n_y = no. outputs, m = no. examples

    :param AL_prime -- the derivative of the last layer's activation output(s) w.r.t. the inputs x: AL' = d(AL)/dX
                >> a numpy array of size (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    :param Y_prime -- the true derivative of the output(s) w.r.t. the inputs x: Y' = d(Y)/dX
               >> a numpy array of shape (n_y, n_x, m) where n_y = no. outputs, n_x = no. inputs, m = no. examples

    :param caches -- list of caches containing every cache of L_model_forward()
              >> a tuple containing {(A_prev, Z, W, b, activation), ..., (A_prev, Z, W, b, activation)}
                                      -------- layer 1 -----------        -------- layer L ----------
                      where
                            A_prev -- activations from previous layer
                                      >> a numpy array of shape (n_prev, m) where n_prev is the no. nodes in layer L-1
                            Z -- input to activation functions for current layer
                                      >> a numpy array of shape (n, m) where n is the no. nodes in layer L
                            W -- weight parameters for current layer
                                 >> a numpy array of shape (n, n_prev)
                            b -- bias parameters for current layer
                                 >> a numpy array of shape (n, 1)

    :param J_caches -- a list of lists containing every cache of L_grads_forward() for each layer (where J stands for Jacobian)
              >> a tuple [ [[...], ..., [...]], ..., [..., (j, Z_prime_j, A_prime_j, G_prime, G_prime_prime), ...], ...]
                            --- layer 1 ------        ------------------ layer l, partial j ---------------------
                      where
                            j -- input variable number (i.e. X1, X2, ...) associated with cache
                                      >> an integer representing the associated input variables (X1, X2, ..., Xj, ...)
                            Z_prime_j -- derivative of Z w.r.t. X_j: Z'_j = d(Z_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l
                            A_prime_j -- derivative of the activation w.r.t. X_j: A'_j = d(A_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l

    :param lambd: float, regularization parameter
    :param gamma: float, gradient-enhancement parameter


    :return grads -- A dictionary with the gradients of the cost function w.r.t. to parameters:
                grads["A" + str(l)] = ...
                grads["W" + str(l)] = ...
                grads["b" + str(l)] = ...
    """
    # Initialize grads
    grads = {}

    # Some quantities needed
    L = len(caches)  # the number of layers
    _, m = AL.shape
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the back propagation
    dA, dA_prime = initialize_back_prop(AL, Y, AL_prime, Y_prime)

    # Loop over each layer
    for l in reversed(range(L)):
        # Get cache
        cache = caches[l]
        J_cache = J_caches[l]

        # Backprop step
        dA, dW, db, dA_prime = linear_activation_backward(
            dA, dA_prime, cache, J_cache, lambd, gamma
        )

        # Store result
        grads["W" + str(l + 1)] = dW
        grads["b" + str(l + 1)] = db

    return grads
