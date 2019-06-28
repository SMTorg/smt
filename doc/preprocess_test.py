"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
        
This package is distributed under New BSD license.
"""

import os, sys
import inspect
import importlib
import contextlib

try:
    from StringIO import StringIO
except:
    from io import StringIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old


def process_test(root, file_name, iline, line):
    file_path = root + "/" + file_name

    embed_num_indent = line.find(".. embed-test")
    if line[:embed_num_indent] != " " * embed_num_indent:
        return line

    include_print_output = (
        "embed-test-print" in line
        or "embed-test-print-plot" in line
        or "embed-test-print-plot" in line
    )
    include_plot_output = (
        "embed-test-plot" in line
        or "embed-test-print-plot" in line
        or "embed-test-print-plot" in line
    )

    split_line = line.replace(" ", "").split(",")
    if len(split_line) != 3 or len(split_line[0].split("::")) != 2:
        raise Exception(
            "Invalid format for embed-test in file {} line {}".format(
                file_path, iline + 1
            )
        )

    py_file_path = split_line[0].split("::")[1]
    class_name = split_line[1]
    method_name = split_line[2][:-1]

    index = len(py_file_path.split("/")[-1])
    py_root = py_file_path[:-index]
    py_file_name = py_file_path[-index:]

    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/" + py_root)
    py_module = importlib.import_module(py_file_name[:-3])

    obj = getattr(py_module, class_name)
    method = getattr(obj, method_name)

    method_lines = inspect.getsource(method).split("\n")
    for imethod_line, method_line in enumerate(method_lines):
        if "def" in method_line and method_name in method_line:
            imethod_line += 1
            break
    method_lines = method_lines[imethod_line:]

    first_line = method_lines[0]
    py_num_indent = first_line.find(first_line.strip())

    for imethod_line, method_line in enumerate(method_lines):
        method_lines[imethod_line] = method_line[py_num_indent:]

    replacement_lines = []

    replacement_lines.append(" " * embed_num_indent + ".. code-block:: python\n")
    replacement_lines.append("\n")
    replacement_lines.extend(
        [
            " " * embed_num_indent + " " * 2 + method_line + "\n"
            for method_line in method_lines
        ]
    )

    if include_print_output:
        joined_method_lines = "\n".join(method_lines)
        with stdoutIO() as s:
            exec(joined_method_lines)

        output_lines = s.getvalue().split("\n")

        if len(output_lines) > 1:
            replacement_lines.append(" " * embed_num_indent + "::\n")
            replacement_lines.append("\n")
            replacement_lines.extend(
                [
                    " " * embed_num_indent + " " * 2 + output_line + "\n"
                    for output_line in output_lines
                ]
            )

    if include_plot_output:
        joined_method_lines = "\n".join(method_lines)
        plt.clf()
        with stdoutIO() as s:
            exec(joined_method_lines)

        abs_plot_name = file_path[:-5] + ".png"
        plt.savefig(abs_plot_name)

        rel_plot_name = file_name[:-5] + ".png"
        replacement_lines.append(
            " " * embed_num_indent + ".. figure:: {}\n".format(rel_plot_name)
        )
        replacement_lines.append(" " * embed_num_indent + "  :scale: 80 %\n")
        replacement_lines.append(" " * embed_num_indent + "  :align: center\n")

    return replacement_lines
