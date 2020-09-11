"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.
"""

import time
import contextlib


class Printer(object):
    """
    Tool for formatting printing and measuring wall times.

    Attributes
    ----------
    active : bool
        If False, the printer is in a state in which printing is suppressed.
    depth : int
        Current level of nesting of the code, affecting the degree of indentation of prints.
    max_print_depth : int
        Maximum depth to print.
    times : dict
        Recorded wall times for operations.
    """

    def __init__(self):
        self.active = False
        self.depth = 1
        self.max_print_depth = 100
        self.times = {}

    def _time(self, key):
        """
        Get the recorded wall time for operation given by key.

        Arguments
        ---------
        key : str
            Unique name of the operation that was previously timed.

        Returns
        -------
        float
            Measured wall time.
        """
        return self.times[key]

    def __call__(self, string="", noindent=False):
        """
        Print the given string.

        Arguments
        ---------
        string : str
            String to print.
        noindent : bool
            If True, suppress any indentation; otherwise, indent based on depth.
        """
        if self.active and self.depth <= self.max_print_depth:
            if noindent:
                print(string)
            else:
                print("   " * self.depth + string)

    def _center(self, string):
        """
        Print string centered based on a line width of 75 characters.

        Arguments
        ---------
        string : str
            String to print.
        """
        pre = " " * int((75 - len(string)) / 2.0)
        self(pre + "%s" % string, noindent=True)

    def _line_break(self):
        """
        Print a line with a width of 75 characters.
        """
        self("_" * 75, noindent=True)
        self()

    def _title(self, title):
        """
        Print a title preceded by a line break.

        Arguments
        ---------
        title : str
            String to print.
        """
        self._line_break()
        self(" " + title, noindent=True)
        self()

    @contextlib.contextmanager
    def _timed_context(self, string=None, key=None):
        """
        Context manager for an operation.

        This context manager does 3 things:
        1. Measures the wall time for the operation.
        2. Increases the depth during the operation so that prints are indented.
        3. Optionally prints a pre-operation and post-operation messages including the time.

        Arguments
        ---------
        string : str or None
            String to print before/after operation if not None.
        key : str
            Name for this operation allowing the measured time to be read later if given.
        """
        if string is not None:
            self(string + " ...")

        start_time = time.time()
        self.depth += 1
        yield
        self.depth -= 1
        stop_time = time.time()

        if string is not None:
            self(string + " - done. Time (sec): %10.7f" % (stop_time - start_time))

        if key is not None:
            if key not in self.times:
                self.times[key] = [stop_time - start_time]
            else:
                self.times[key].append(stop_time - start_time)
