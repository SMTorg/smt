"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
        
This package is distributed under New BSD license.
"""

try:
    import cPickle as pickle
except:
    import pickle
import hashlib
import contextlib


@contextlib.contextmanager
def cached_operation(inputs_dict, data_dir, desc=""):
    """
    Context manager for an operation that may be cached.

    Arguments
    ---------
    inputs_dict : dict
        Dictionary containing the inputs of the operation.
    data_dir : None or str
        Directory containing the cached data files; if None, do not load or save.
    desc : str
        Optional descriptive prefix for the filename.

    Yields
    ------
    outputs_dict : dict
        Dictionary containing the outputs of the operation.
    """
    checksum = _caching_checksum(inputs_dict)
    filename = "%s/%s_%s.dat" % (data_dir, desc, checksum)
    try:
        with open(filename, "rb") as f:
            outputs_dict = pickle.load(f)
        load_successful = True
    except:
        outputs_dict = {}
        load_successful = False

    yield outputs_dict

    if not load_successful and data_dir:
        with open(filename, "wb") as f:
            pickle.dump(outputs_dict, f)


def _caching_checksum(obj):
    """
    Compute the hex string checksum of the given object.

    Arguments
    ---------
    obj : object
        Object to compute the checksum for; normally a dictionary.

    Returns
    -------
    str
        Hexadecimal string checksum that was computed.
    """
    try:
        tmp = obj["self"].printer
        obj["self"].printer = None
    except:
        pass

    self_pkl = pickle.dumps(obj)
    checksum = hashlib.md5(self_pkl).hexdigest()

    try:
        obj["self"].printer = tmp
    except:
        pass

    return checksum
