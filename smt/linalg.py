"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
"""

from __future__ import print_function
import numpy as np
import scipy.sparse
import six
from six.moves import range

from smt.linear_solvers import LinearSolver, DirectSolver, KrylovSolver
from smt.linear_solvers import StationarySolver, MultigridSolver


def assemble_sparse_mtx(block_names, block_sizes, sub_mtx_dict, sub_rhs_dict):
    name2ind = {}
    for ind, name in enumerate(block_names):
        name2ind[name] = ind

    sub_mtx_list = [[None for name in block_names] for name in block_names]
    for (row_name, col_name), sub_mtx in six.iteritems(sub_mtx_dict):
        row_ind = name2ind[row_name]
        col_ind = name2ind[col_name]
        sub_mtx_list[row_ind][col_ind] = sub_mtx

    mtx = scipy.sparse.bmat(sub_mtx_list, format='csc')

    rhs = np.zeros((np.sum(block_sizes), sub_rhs_dict.values()[0].shape[1]))
    for name, sub_rhs in six.iteritems(sub_rhs_dict):
        ind = name2ind[name]
        ind1 = np.sum(block_sizes[:ind], dtype=int)
        ind2 = np.sum(block_sizes[:ind+1], dtype=int)
        rhs[ind1:ind2, :] = sub_rhs

    return mtx, rhs
