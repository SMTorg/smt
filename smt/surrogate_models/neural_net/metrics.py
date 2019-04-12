"""
G R A D I E N T - E N H A N C E D   N E U R A L   N E T W O R K S  (G E N N)

Author: Steven H. Berguin <steven.berguin@gtri.gatech.edu>

This package is distributed under New BSD license.
"""

import numpy as np


def rsquare(Y_pred, Y_true):
    """
    Compute R-square for a single response.

    NOTE: If you have more than one response, then you'll either have to modify this method to handle many responses at
          once or wrap a for loop around it (i.e. treat one response at a time).

    Arguments:
    Y_pred -- predictions,  numpy array of shape (K, m) where n_y = no. of outputs, m = no. of examples
    Y_true -- true values,  numpy array of shape (K, m) where n_y = no. of outputs, m = no. of examples

    Return:
    R2 -- the R-square value,  numpy array of shape (K, 1)
    """
    epsilon = 1e-8  # small number to avoid division by zero
    Y_bar = np.mean(Y_true)
    SSE = np.sum(np.square(Y_pred - Y_true), axis=1)
    SSTO = np.sum(np.square(Y_true - Y_bar) + epsilon, axis=1)
    R2 = 1 - SSE / SSTO
    return R2
