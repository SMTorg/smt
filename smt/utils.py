"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
"""

from __future__ import print_function
import cPickle as pickle
import hashlib
import time


class Timer(object):

    def __init__(self, print_status=True):
        self.print_status = print_status

        self.raw_times = {}
        self.elapsed_times = {}

    def _start(self, key, desc=None):
        self.raw_times[key] = time.time()

        if self.print_status and desc is not None:
            print('   %s ... ' % desc)

    def _stop(self, key, desc=None, print_done=False):
        self.elapsed_times[key] = time.time() - self.raw_times[key]
        self.raw_times.pop(key)

        if self.print_status and desc is not None:
            self._print(key, desc)

        if self.print_status and print_done:
            print('   Done. Time (sec) : %10.7f' % self.elapsed_times[key])
            print()

    def _print(self, key, desc, div_factor=1.0):
        elapsed_time = self.elapsed_times[key] / div_factor

        if self.print_status:
            print('   %-14s : %10.7f' % (desc, elapsed_time))
            print()


class OptionsDictionary(object):

    def __init__(self, options):
        self._dict = options

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def add(self, name, default=None, values=None, type_=None, desc=''):
        if values is not None:
            if default is not None:
                assert default in values, \
                    'Option %s: default %s not in %s' % (name, default, values)
            if name in self:
                assert self[name] in values, \
                    'Option %s: value %s not in %s' % (name, self[name], values)

        if type_ is not None:
            if default is not None:
                assert isinstance(default, type_), \
                    'Option %s: default %s should be type %s' % (name, default, type_)
            if name in self:
                assert isinstance(self[name], type_), \
                    'Option %s: default %s should be type %s' % (name, self[name], type_)

        if name not in self:
            assert default is not None, 'Required option %s not given' % name
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
    tmp = sm.timer

    sm.timer = None
    checksum = _caching_checksum(sm)
    sm.timer = tmp
    return checksum
