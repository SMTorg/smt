"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
"""

from __future__ import print_function
import cPickle as pickle
import hashlib
import time
import contextlib


class Printer(object):

    def __init__(self):
        self.active = False
        self.depth = 1
        self.max_print_depth = 100
        self.times = {}

    def _time(self, key):
        return self.times[key]

    def __call__(self, string='', noindent=False):
        if self.active and self.depth <= self.max_print_depth:
            if noindent:
                print(string)
            else:
                print('   ' * self.depth + string)

    def _center(self, string):
        pre = ' ' * int((75 - len(string))/2.0)
        self(pre + '%s' % string, noindent=True)

    def _line_break(self):
        self('_' * 75, noindent=True)
        self()

    def _title(self, title):
        self._line_break()
        self(' ' + title, noindent=True)
        self()

    @contextlib.contextmanager
    def _timed_context(self, string=None, key=None):
        if string is not None:
            self(string + ' ...')
            # self()

        start_time = time.time()
        self.depth += 1
        yield
        self.depth -= 1
        stop_time = time.time()

        if string is not None:
            self(string + ' - done. Time (sec): %10.7f' % (stop_time - start_time))
            # self()

        if key is not None:
            self.times[key] = stop_time - start_time


class OptionsDictionary(object):

    def __init__(self):
        self._dict = {}
        self._declared_values = {}
        self._declared_types = {}

    def __getitem__(self, name):
        return self._dict[name]

    def __setitem__(self, name, value):
        assert name in self._declared_values, 'Option %s has not been declared' % name
        self._assert_valid(name, value)
        self._dict[name] = value

    def __contains__(self, key):
        return key in self._dict

    def _assert_valid(self, name, value):
        values = self._declared_values[name]
        types = self._declared_types[name]

        if values is not None and types is not None:
            assert value in values or isinstance(value, types), \
                'Option %s: value and type of %s are both invalid - ' \
                + 'value must be %s or type must be %s' % (name, value, values, types)
        elif values is not None:
            assert value in values, \
                'Option %s: value %s is invalid - must be %s' % (name, value, values)
        elif types is not None:
            assert isinstance(value, types), \
                'Option %s: type of %s is invalid - must be %s' % (name, value, types)

    def update(self, dict_):
        for name in dict_:
            self[name] = dict_[name]

    def declare(self, name, default=None, values=None, types=None, desc=''):
        self._declared_values[name] = values
        self._declared_types[name] = types

        if default is not None:
            self._assert_valid(name, default)

        self._dict[name] = default

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

def _caching_checksum_sm(sm):
    tmp = sm.printer

    sm.printer = None
    checksum = _caching_checksum(sm)
    sm.printer = tmp
    return checksum

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
