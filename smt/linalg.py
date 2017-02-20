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

def solve_sparse_system(mtx, rhs, sol, sm_options, print_global, mg_ops=[]):
    if sm_options['solver'] == 'direct':
        solver = DirectSolver()
    elif sm_options['solver'] == 'krylov':
        solver = KrylovSolver(solver='gmres', pc='nopc', print_solve=True,
                              ilimit=100, atol=1e-15, rtol=1e-15)
    elif sm_options['solver'] == 'gs' or sm_options['solver'] == 'jacobi':
        solver = StationarySolver(solver=sm_options['solver'], damping=1.0, print_solve=True,
                                  ilimit=100, atol=1e-15, rtol=1e-15)
    elif sm_options['solver'] == 'mg':
        solver = MultigridSolver(mg_ops=mg_ops, print_solve=True)
    elif isinstance(sm_options['solver'], LinearSolver):
        solver = sm_options['solver']

    solver._initialize(mtx, print_global)

    for ind_y in range(rhs.shape[1]):
        solver.timer._start('convergence')
        solver._solve(rhs[:, ind_y], sol[:, ind_y], print_global)
        solver.timer._stop('convergence')
        solver.printer._total_time('Total solver convergence time (sec)',
                                   solver.timer['convergence'])
