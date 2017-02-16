"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
"""

from __future__ import print_function
import cPickle as pickle
import hashlib
import time


class Timer(object):

    def __init__(self):
        self.raw_times = {}
        self.elapsed_times = {}

    def __getitem__(self, key):
        return self.elapsed_times[key]

    def _start(self, key):
        if key in self.raw_times:
            raise RuntimeError('The timer has already been started for key %s' % key)

        self.raw_times[key] = time.time()

    def _stop(self, key):
        start_time = self.raw_times.pop(key)
        stop_time = time.time()
        self.elapsed_times[key] = stop_time - start_time


class Printer(object):

    def __init__(self, active=False):
        self.active = active

    def __call__(self, string=''):
        if self.active:
            print(string)

    def _center(self, string):
        pre = ' ' * int((75 - len(string))/2.0)
        self(pre + '%s' % string)

    def _line_break(self):
        self('_' * 75)
        self()

    def _title(self, title):
        self._line_break()
        self(' ' + title)
        self()

    def _operation(self, string):
        self('   %s ... ' % string)

    def _done_time(self, elapsed_time):
        self('   Done. Time (sec) : %10.7f' % elapsed_time)
        self()

    def _total_time(self, string, elapsed_time):
        self('   %-14s : %10.7f' % (string, elapsed_time))
        self()


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
