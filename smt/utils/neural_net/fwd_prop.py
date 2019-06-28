"""
G R A D I E N T - E N H A N C E D   N E U R A L   N E T W O R K S  (G E N N)

Author: Steven H. Berguin <steven.berguin@gtri.gatech.edu>

This package is distributed under New BSD license.
"""

import numpy as np


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement forward propagation for one layer.

    Arguments:
    :param A_prev -- activations from previous layer
        >> numpy array of size (n[l-1], 1) where n[l-1] = no. nodes in previous layer

    :param W -- weights associated with current layer l
        >> numpy array of size (n_l, n[l-1]) where n_l = no. nodes in current layer

    :param b -- biases associated with current layer
        >> numpy array of size (n_l, 1)

    :param activation -- activation function for this layer

    Return:
    :return A -- a vector of post-activation values of current layer
    :return cache -- parameters that can be used in other functions:
            >> a tuple (A_prev, Z, W, b)    where       A_prev -- a numpy array of shape (n[l-1], m) containing previous
                                                                  layer post-activation values where:
                                                                            n[l-1] = no. nodes in previous layer
                                                                            m = no. of training examples
                                                        Z -- a numpy array of shape (n[l], m) containing linear forward
                                                             values where n_l = no. nodes in current layer
                                                        W -- a numpy array of shape (n[l], n[l-1]) containing weights of
                                                             the current layer
                                                        b -- a numpy array of shape (n[l], 1) containing biases of
                                                             the current layer
    """
    Z = np.dot(W, A_prev) + b
    A = activation.evaluate(Z)
    cache = (A_prev, Z, W, b, activation)

    return A, cache


def L_model_forward(X, parameters, activations):
    """
    Implements forward propagation for the entire neural network.

    Arguments:
    :param X -- data, numpy array of shape (n_x, m) where n_x = no. of inputs, m = no. of training examples

    :param parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}

    :param activations -- a list of Activation objective (one for each layer)

    :return AL -- last post-activation value
        >> numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of training examples

    :return caches -- a list of tuples containing every cache of linear_activation_forward()
                Note: there are L-1 of them, indexed from 0 to L-2
            >> [(...), (A_prev, Z, W, b), (...)] where  A_prev -- a numpy array of shape (n[l-1], m) containing previous
                                                                  layer post-activation values where:
                                                                            n[l-1] = no. nodes in previous layer
                                                                            m = no. of training examples
                                                        Z -- a numpy array of shape (n[l], m) containing linear forward
                                                             values where n_l = no. nodes in current layer
                                                        W -- a numpy array of shape (n[l], n[l-1]) containing weights of
                                                             the current layer
                                                        b -- a numpy array of shape (n[l], 1) containing biases of
                                                             the current layer
    """
    caches = []
    A = X
    L = len(
        activations
    )  # number of layers in the network (doesn't include input layer)
    for l in range(1, L + 1):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = linear_activation_forward(
            A_prev, W, b, activation=activations[l - 1]
        )
        caches.append(cache)

    return A, caches


def L_grads_forward(X, parameters, activations):
    """
    Compute the gradient of the neural network evaluated at X.

    Argument:
    :param X -- data, numpy array of shape (n_x, m) where n_x = no. of inputs, m = no. of training examples

    :param parameters -- parameters of the neural network as defined in initialize_parameters()
                        >> a dictionary containing: {"W1": a numpy array of shape (n[1], n[0])}    N.B. n[0] = n_x
                                                    {"W2": a numpy array of shape (n[2], n[1])}
                                                    {"W3": a numpy array of shape (n[3], n[2])}
                                                    ...
                                                    {"WL": a numpy array of shape (n[L], n[L-1])}  N.B. n[L] = n_y
                                                    {"b1": a numpy array of shape (n[1], 1)}
                                                    {"b2": a numpy array of shape (n[2], 1)}
                                                    {"b3": a numpy array of shape (n[3], 1)}
                                                    ...
                                                    {"bL": a numpy array of shape (n[L], 1)}

    :param activations -- a list of Activation objective (one for each layer)

    :return JL -- numpy array of size (n_y, n_x, m) containing the Jacobian of w.r.t. X where n_y = no. of outputs

    :return J_caches -- list of caches containing every cache of L_grads_forward() where J stands for Jacobian
              >> a list containing [..., (j, Z_prime_j, A_prime_j, G_prime, G_prime_prime), ...]
                                          ------------------ input j --------------------
                      where
                            j -- input variable number (i.e. X1, X2, ...) associated with cache
                                      >> an integer representing the associated input variables (X1, X2, ..., Xj, ...)
                            Z_prime_j -- derivative of Z w.r.t. X_j: Z'_j = d(Z_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l
                            A_prime_j -- derivative of the activation w.r.t. X_j: A'_j = d(A_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the no. nodes in layer l
    """
    J_caches = []

    # Dimensions
    L = len(activations)  # number of layers in network
    n_y = parameters["W" + str(L)].shape[0]  # number of outputs
    try:
        n_x, m = X.shape  # number of inputs, number of examples
    except ValueError:
        n_x = X.size
        m = 1
        X = X.reshape(n_x, m)

    # Initialize Jacobian for layer 0 (one example)
    I = np.eye(n_x, dtype=float)

    # Initialize Jacobian for layer 0 (all m examples)
    J0 = np.repeat(I.reshape((n_x, n_x, 1)), m, axis=2)

    # Initialize Jacobian for last layer
    JL = np.zeros((n_y, n_x, m))

    # Initialize caches
    for l in range(0, L):
        J_caches.append([])

    # Loop over partials
    for j in range(0, n_x):

        # Initialize (first layer)
        A = np.copy(X).reshape(n_x, m)
        A_prime_j = J0[:, j, :]

        # Loop over layers
        for l in range(1, L + 1):

            # Previous layer
            A_prev = A
            A_prime_j_prev = A_prime_j

            # Get parameters for this layer
            W = parameters["W" + str(l)]
            b = parameters["b" + str(l)]
            activation = activations[l - 1]

            # Linear
            Z = np.dot(W, A_prev) + b

            # The following is not needed here, but it is needed later, during backprop.
            # We will thus compute it here and store it as a cache for later use.
            Z_prime_j = np.dot(W, A_prime_j_prev)

            # Activation
            A = activation.evaluate(Z)
            G_prime = activation.first_derivative(Z)

            # Current layer output gradient
            A_prime_j = G_prime * np.dot(W, A_prime_j_prev)

            # Store cache
            J_caches[l - 1].append((j, Z_prime_j, A_prime_j_prev))

        # Store partial
        JL[:, j, :] = A_prime_j

    if m == 1:
        JL = JL[:, :, 0]

    return JL, J_caches
