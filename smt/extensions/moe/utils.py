"""
This file contains useful functions for manipulating arrays and some objective functions
"""
import math
import numpy as np
from numpy import float64
from numpy import fabs


class UncorrectInputDimension(RuntimeError):
    "A class for input dimension errors"
    pass


def cut_list(list_, n):
    """
    Cut a list_ in n lists with the same number of samples
    Parameters:
    -----------
    - list_ : Array_like
    The list_ to cut
    - n : int
    Number of lists needed
    Return:
    -------
    - n_list : array_like
    List of n lists
    """
    n_list = []
    for i in range(n):
        n_list.append([])
    i = 0
    length = len(list_)
    while i < length:
        j = 0
        while i < length and j < n:
            n_list[j].append(list_[i])
            i = i + 1
            j = j + 1
    return n_list


def concat_except(list_, n):
    """
    Concatenate a list_ of N lists except the n_th list
    Parameters:
    -----------
    - list_ : Array_like
    A list_ of N lists
    - n : int
    The number of the list to eliminate
    Return:
    -------
    - nlist : array_like
    The concatenated list
    """
    nlist = []
    for i in range(len(list_)):
        if i != n:
            nlist = nlist + list_[i]
    return nlist


def map_2d_space(x_max=None, x_min=None, num=None):
    """
    Map a 2D-space
    Optional
    --------
    - x_max : maximum
    [1,1] if None
    - x_min : minimum
    [0,0] if None
    - num : Points number by dimension
    50 if None
    Returns
    -------
    mapping_array : 2D array mapping the space
    """
    if x_max is None:
        x_max = [1, 1]
    if x_min is None:
        x_min = [0, 0]
    if num is None:
        num = 50

    x_array = np.linspace(x_min[0], x_max[0], num)
    y_array = np.linspace(x_min[1], x_max[1], num)
    mapping_array = []

    for i in range(len(x_array)):
        for j in range(len(y_array)):
            mapping_array.append([x_array[i], y_array[j]])

    return np.array(mapping_array)


def norm1(x):
    """
    This function is the norm1 function
    Input:
    --------
    x : array_like
    Output:
    --------
    y : array_like
    Values of norm1 function for each x
    """
    y = []
    for i in range(len(x)):
        if isinstance(x[i], float64) or isinstance(x[i], float):
            y.append(math.fabs(x[i]))
        else:
            y.append(sum(fabs(x[i])))
    return y


def grad_norm1_2d(x):
    """
    This function computes the jacobian of the norm1 function
    Input:
    --------
    x : 2D array_like
    Output:
    --------
    y : array_like
    Values of the jacobian of the norm1 function for each x
    """
    if x.shape[1] != 2:
        raise UncorrectInputDimension()

    y = []
    for i in range(len(x)):
        new_point = []
        for j in range(len(x[i])):
            if x[i][j] < 0:
                new_point.append(-1)
            else:
                new_point.append(1)
        y.append(new_point)
    return np.array(y)


def branin(x):
    """
    This function is the Branin function
    Input:
    --------
    x : 2D array_like
    Output
    --------
    y : array_like
    values of Branin function for each x
    """
    if x.shape[1] != 2:
        raise UncorrectInputDimension()

    x1 = 15 * x[:, 0] - 5
    x2 = 15 * x[:, 1]
    return (x2 - 5.1 / (4. * (np.pi)**2) * x1**2 + 5. / np.pi * x1 - 6)**2 + \
        10. * (1. - 1. / (8. * np.pi)) * np.cos(x1) + 10


def grad_branin(x):
    """
    This function calculates the jacobian of the Branin function
    Input:
    --------
    x : 2D array_like
    Output:
    --------
    y : array_like
    Values of the jacobian of Branin function for each x
    """
    if x.shape[1] != 2:
        raise UncorrectInputDimension()

    x1 = 15 * x[:, 0] - 5
    x2 = 15 * x[:, 1]

    u = (x2 - 5.1 / (4. * (np.pi)**2) * x1**2 + 5. / np.pi * x1 - 6)
    u1 = (-5.1 / (2. * (np.pi)**2) * x1 + 5. / np.pi)
    v1 = - 10. * (1. - 1. / (8. * np.pi)) * np.sin(x1)

    y = []
    y.append(2 * 15 * u1 * u + 15 * v1)
    y.append(2 * 15 * u)

    return np.array(y).T


def sum_x(n):
    """
    Compute the sum from 0 to n
    Parameters:
    -----------
    - n : int
    Last integer to sum
    Return:
    -------
    Sum of the first integer until n
    """
    if n > 0:
        return n * (n + 1) / 2
    else:
        return 0
