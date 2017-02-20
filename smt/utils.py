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
    tmp = sm.timer

    sm.timer = None
    checksum = _caching_checksum(sm)
    sm.timer = tmp
    return checksum
