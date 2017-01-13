from __future__ import print_function
import numpy
import scipy.sparse
import scipy.sparse.linalg
import six
from six.moves import range


def assemble_sparse_mtx(block_names, block_sizes, sub_mtx_dict, sub_rhs_dict):
    name2ind = {}
    for ind, name in enumerate(block_names):
        name2ind[name] = ind

    sub_mtx_list = [[None for name in block_names] for name in block_names]
    for (row_name, col_name), sub_mtx in six.iteritems(sub_mtx_dict):
        row_ind = name2ind[row_name]
        col_ind = name2ind[col_name]
        sub_mtx_list[row_ind, col_ind] = sub_mtx

    mtx = scipy.sparse.bmat(sub_mtx_list, format='csc')

    rhs = numpy.zeros(numpy.sum(block_sizes))
    for name, sub_rhs in six.iteritems(sub_rhs_dict):
        ind = name2ind[name]
        ind1 = numpy.sum(block_sizes[:ind])
        ind2 = numpy.sum(block_sizes[:ind+1])
        rhs[ind1:ind2] = sub_rhs

    return mtx, rhs

def solve_sparse_system(mtx, rhs, sol, solver_options):
    if solver_options['pc']:
        pc = scipy.sparse.linalg.spilu(pc, drop_tol=1e-16,
                                       fill_factor=10, drop_rule='basic')
        pc_op = scipy.sparse.linalg.LinearOperator((nmat, nmat), matvec=pc.solve)
    elif solver_options['pc']:
        pc = scipy.sparse.linalg.splu(pc)
        pc_op = scipy.sparse.linalg.LinearOperator((nmat, nmat), matvec=pc.solve)
    elif solver_options['pc'] == 'nopc':
        pc_op = None

    counter = 0
    if solver_options['print']:
        def callback(res):
            print(counter, numpy.linalg.norm(res))
            counter += 1
    else:
        callback = lambda res: None

    for irhs in range(rhs.shape[1]):
        counter = 0
        sol[:, k], info = scipy.sparse.linalg.gmres(
            mtx, rhs[:, k], M=pc_op, callback=callback,
            maxiter=solver_options['pc']['ilimit'],
            tol=solver_options['pc']['atol'],
        )
