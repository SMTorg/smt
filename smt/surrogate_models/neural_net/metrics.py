"""
G R A D I E N T - E N H A N C E D   N E U R A L   N E T W O R K S  (G E N N)

Author: Steven H. Berguin <steven.berguin@gtri.gatech.edu>

This package is distributed under New BSD license.
"""

import numpy as np


def compute_precision(Y_pred, Y_true):
    """
    Compute precision = True positives / Total Number of Predicted Positives
                      = True positives / (True Positives + False Positives)

    NOTE: This method applies to binary classification only!

    Arguments:
    Y_pred -- predictions,  numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of examples
    Y_true -- true values,  numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of examples

    Return:
    P -- precision, numpy array of (n_y, 1)
        >> P is a number between 0 and 1 where 0 is bad and 1 is good
    """
    true_positives = np.sum(((Y_pred + Y_true) == 2).astype(float), axis=1, keepdims=True)
    false_positives = np.sum(((Y_pred - Y_true) == 1).astype(float), axis=1, keepdims=True)
    if true_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)
    return precision


def compute_recall(Y_pred, Y_true):
    """
    Compute recall = True positives / Total Number of Actual Positives
                   = True positives / (True Positives + False Negatives)

    NOTE: This method applies to classification only!

    Arguments:
    Y_pred -- predictions,  numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of examples
    Y_true -- true values,  numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of examples

    Return:
    R -- recall, numpy array of (n_y, 1)
        >> R is a number between 0 and 1 where 0 is bad and 1 is good
    """
    true_positives = np.sum(((Y_pred + Y_true) == 2).astype(float), axis=1, keepdims=True)
    false_negatives = np.sum(((Y_true - Y_pred) == 1).astype(float), axis=1, keepdims=True)
    if true_positives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)
    return recall


def compute_Fscore(Y_pred, Y_true):
    """
    Compute F-scoare = 2*P*R / (P + R) where P = precision
                                             R = recall

    NOTE: This method applies to classification only!

    Arguments:
    Y_pred -- predictions,  numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of examples
    Y_true -- true values,  numpy array of shape (n_y, m) where n_y = no. of outputs, m = no. of examples

    Return:
    F -- F-score, numpy array of (n_y, 1)
        >> F is a number between 0 and 1 where 0 is bad and 1 is good
    """
    P = compute_precision(Y_pred, Y_true)
    R = compute_recall(Y_pred, Y_true)
    if (P + R) == 0:
        F = 0
    else:
        F = 2 * P * R / (P + R)
    return F


def goodness_fit_regression(Y_pred, Y_true):
    """
    Compute goodness of fit metrics: R2, std(error), avg(error).

    Note: these metrics only apply to regression

    Arguments:
    Y_pred -- numpy array of size (K, m) where K = num outputs, n = num examples
    Y_true -- numpy array of size (K, m) where K = num outputs, m = num examples

    Return:
    R2 -- float, RSquare value
    sig -- numpy array of shape (K, 1), standard deviation of error
    mu -- numpy array of shape (K, 1), avg value of error expressed
    """
    K = Y_true.shape[0]
    R2 = rsquare(Y_pred, Y_true)
    sig = np.std(Y_pred - Y_true)
    mu = np.mean(Y_pred - Y_true)

    return R2.reshape(K, 1), sig.reshape(K, 1), mu.reshape(K, 1)


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
