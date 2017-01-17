from __future__ import print_function
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import cPickle as pickle
import hashlib
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
        sub_mtx_list[row_ind][col_ind] = sub_mtx

    mtx = scipy.sparse.bmat(sub_mtx_list, format='csc')

    rhs = np.zeros((np.sum(block_sizes), sub_rhs_dict.values()[0].shape[1]))
    for name, sub_rhs in six.iteritems(sub_rhs_dict):
        ind = name2ind[name]
        ind1 = np.sum(block_sizes[:ind], dtype=int)
        ind2 = np.sum(block_sizes[:ind+1], dtype=int)
        rhs[ind1:ind2, :] = sub_rhs

    return mtx, rhs

def solve_sparse_system(mtx, rhs, sol, sm_options, printf_options):
    if sm_options['solver_pc'] == 'ilu':
        pc = scipy.sparse.linalg.spilu(mtx, drop_tol=1e-16,
                                       fill_factor=10, drop_rule='basic')
        pc_op = scipy.sparse.linalg.LinearOperator(mtx.shape, matvec=pc.solve)
    elif sm_options['solver_pc'] == 'lu':
        pc = scipy.sparse.linalg.splu(mtx)
        pc_op = scipy.sparse.linalg.LinearOperator(mtx.shape, matvec=pc.solve)
    elif sm_options['solver_pc'] == 'nopc':
        pc_op = None

    class Callback(object):

        def __init__(self, print_):
            self.counter = 0
            self.iy = 0
            self.print_ = print_
            if self.print_:
                print('   Solver output (preconditioner: %s)' % sm_options['solver_pc'])
                print('   %3s %3s Residual' % (' iy', 'Itn'))

        def __call__(self, res):
            if self.print_:
                print('   %3i %3i %.9g' %
                    (self.iy, self.counter, np.linalg.norm(res)))
            self.counter += 1

    cb = Callback(printf_options['global'] and printf_options['solver'])

    for irhs in range(rhs.shape[1]):
        cb.counter = 0
        cb.iy = irhs
        sol[:, irhs], info = scipy.sparse.linalg.gmres(
            mtx, rhs[:, irhs], M=pc_op, callback=cb,
            maxiter=sm_options['solver_ilimit'],
            tol=sm_options['solver_atol'],
        )

def _caching_load(filename, checksum):
    try:
        save_pkl = pickle.load(open(filename, 'r'))

        if checksum == save_pkl['checksum']:
            return True, save_pkl['data']
        else:
            return False, None
    except:
        return False, None

def _caching_save(filename, checksum, data):
    save_dict = {
        'checksum': checksum,
        'data': data,
    }
    pickle.dump(save_dict, open(filename, 'w'))

def _caching_checksum(obj):
    self_pkl = pickle.dumps(obj)
    checksum = hashlib.md5(self_pkl).hexdigest()
    return checksum
