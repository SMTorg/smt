import os, sys
import inspect
import importlib
import contextlib
from io import StringIO


@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old


def process_test(file_path, iline, line):
    embed_num_indent = line.find('.. embed-test')
    if line[:embed_num_indent] != ' ' * embed_num_indent:
        return line

    include_print_output = (
        'embed-test-print' in line or
        'embed-test-print-plot' in line or
        'embed-test-plot-print' in line
    )
    include_plot_output = (
        'embed-test-plot' in line or
        'embed-test-print-plot' in line or
        'embed-test-plot-print' in line
    )

    split_line = line.replace(' ', '').split(',')
    if len(split_line) != 3 or len(split_line[0].split('::')) != 2:
        raise Exception('Invalid format for embed-test in file {} line {}'.format(
            file_path, iline + 1))

    py_file_path = split_line[0].split('::')[1]
    class_name = split_line[1]
    method_name = split_line[2][:-1]

    index = len(py_file_path.split('/')[-1])
    py_root = py_file_path[:-index]
    py_file_name = py_file_path[-index:]

    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/' + py_root)
    py_module = importlib.import_module(py_file_name[:-3])

    obj = getattr(py_module, class_name)()
    method = getattr(obj, method_name)

    method_lines = inspect.getsource(method).split('\n')
    for imethod_line, method_line in enumerate(method_lines):
        if 'def' in method_line and method_name in method_line:
            imethod_line += 1
            break
    method_lines = method_lines[imethod_line:]

    first_line = method_lines[0]
    py_num_indent = first_line.find(first_line.strip())

    for imethod_line, method_line in enumerate(method_lines):
        method_lines[imethod_line] = method_line[py_num_indent:]

    replacement_lines = []

    replacement_lines.append(' ' * embed_num_indent + '.. code-block:: python\n')
    replacement_lines.append('\n')
    replacement_lines.extend([
        ' ' * embed_num_indent + ' ' * 2 + method_line + '\n'
        for method_line in method_lines
    ])

    if include_print_output:
        with stdoutIO() as s:
            joined_method_lines = '\n'.join(method_lines)
            exec(joined_method_lines)
            # for method_line in method_lines:
            #     exec(method_line)
        output_lines = s.getvalue().split('\n')

        if len(output_lines) > 1:
            replacement_lines.append(' ' * embed_num_indent + '::\n')
            replacement_lines.append('\n')
            replacement_lines.extend([
                ' ' * embed_num_indent + ' ' * 2 + output_line + '\n'
                for output_line in output_lines
            ])

    if include_plot_output:
        replacement_lines.append(' ' * embed_num_indent + '.. plot::\n')
        replacement_lines.append('\n')
        replacement_lines.extend([
            ' ' * embed_num_indent + ' ' * 2 + method_line + '\n'
            for method_line in method_lines
        ])

    return replacement_lines
