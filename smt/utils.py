from __future__ import print_function
import numpy
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

    rhs = numpy.zeros((numpy.sum(block_sizes), sub_rhs_dict.values()[0].shape[1]))
    for name, sub_rhs in six.iteritems(sub_rhs_dict):
        ind = name2ind[name]
        ind1 = numpy.sum(block_sizes[:ind], dtype=int)
        ind2 = numpy.sum(block_sizes[:ind+1], dtype=int)
        rhs[ind1:ind2, :] = sub_rhs

    return mtx, rhs

def solve_sparse_system(mtx, rhs, sol, solver_options):
    if solver_options['pc'] == 'ilu':
        pc = scipy.sparse.linalg.spilu(mtx, drop_tol=1e-16,
                                       fill_factor=10, drop_rule='basic')
        pc_op = scipy.sparse.linalg.LinearOperator(mtx.shape, matvec=pc.solve)
    elif solver_options['pc'] == 'lu':
        pc = scipy.sparse.linalg.splu(mtx)
        pc_op = scipy.sparse.linalg.LinearOperator(mtx.shape, matvec=pc.solve)
    elif solver_options['pc'] == 'nopc':
        pc_op = None

    class Callback(object):

        def __init__(self):
            self.counter = 0

        def __call__(self, res):
            print(self.counter, numpy.linalg.norm(res))
            self.counter += 1

    if solver_options['print']:
        cb = Callback()
        Callback.counter = 0
    else:
        cb = lambda res: None

    for irhs in range(rhs.shape[1]):
        counter = 0
        sol[:, irhs], info = scipy.sparse.linalg.gmres(
            mtx, rhs[:, irhs], M=pc_op, callback=cb,
            maxiter=solver_options['ilimit'],
            tol=solver_options['atol'],
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
