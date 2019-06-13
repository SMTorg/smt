"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
        
This package is distributed under New BSD license.
"""


class OptionsDictionary(object):
    """
    Generalization of the dictionary that allows for declaring keys.

    Attributes
    ----------
    _dict : dict
        Dictionary of option values keyed by option names.
    _declared_entries : dict
        Dictionary of declared entries.
    """

    def __init__(self):
        self._dict = {}
        self._declared_entries = {}

    def clone(self):
        """
        Return a clone of this object.

        Returns
        -------
        OptionsDictionary
            Deep-copied clone.
        """
        clone = self.__class__()
        clone._dict = dict(self._dict)
        clone._declared_entries = dict(self._declared_entries)
        return clone

    def __getitem__(self, name):
        """
        Get an option that was previously declared and optionally set.

        Arguments
        ---------
        name : str
            The name of the option.

        Returns
        -------
        object
            Value of the option.
        """
        return self._dict[name]

    def __setitem__(self, name, value):
        """
        Set an option that was previously declared.

        The value argument must be valid, which means it must satisfy the following:
        1. If values and not types was given when declaring, value must be in values.
        2. If types and not values was given when declaring, type(value) must be in types.
        3. If values and types were given when declaring, either of the above must be true.

        Arguments
        ---------
        name : str
            The name of the option.
        value : object
            The value to set.
        """
        assert name in self._declared_entries, "Option %s has not been declared" % name
        self._assert_valid(name, value)
        self._dict[name] = value

    def __contains__(self, key):
        return key in self._dict

    def is_declared(self, key):
        return key in self._declared_entries

    def _assert_valid(self, name, value):
        values = self._declared_entries[name]["values"]
        types = self._declared_entries[name]["types"]

        if values is not None and types is not None:
            assert value in values or isinstance(
                value, types
            ), "Option %s: value and type of %s are both invalid - " % (
                name,
                value,
            ) + "value must be %s or type must be %s" % (
                values,
                types,
            )
        elif values is not None:
            assert value in values, "Option %s: value %s is invalid - must be %s" % (
                name,
                value,
                values,
            )
        elif types is not None:
            assert isinstance(
                value, types
            ), "Option %s: type of %s is invalid - must be %s" % (name, value, types)

    def update(self, dict_):
        """
        Loop over and set all the entries in the given dictionary into self.

        Arguments
        ---------
        dict_ : dict
            The given dictionary. All keys must have been declared.
        """
        for name in dict_:
            self[name] = dict_[name]

    def declare(self, name, default=None, values=None, types=None, desc=""):
        """
        Declare an option.

        The value of the option must satisfy the following:
        1. If values and not types was given when declaring, value must be in values.
        2. If types and not values was given when declaring, type(value) must be in types.
        3. If values and types were given when declaring, either of the above must be true.

        Arguments
        ---------
        name : str
            Name of the option.
        default : object
            Optional default value that must be valid under the above 3 conditions.
        values : list
            Optional list of acceptable option values.
        types : type or list of types
            Optional list of acceptable option types.
        desc : str
            Optional description of the option.
        """
        self._declared_entries[name] = {
            "values": values,
            "types": types,
            "default": default,
            "desc": desc,
        }

        if default is not None:
            self._assert_valid(name, default)

        self._dict[name] = default
