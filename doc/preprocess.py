"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>
        
This package is distributed under New BSD license.
"""

import os

from preprocess_test import process_test
from preprocess_options import process_options


def get_file_paths():
    file_paths_list = []
    file_paths_list_rstx = []
    file_paths_list_py = []

    for root, dirs, files in os.walk("."):
        for file_name in files:
            file_paths_list.append((root, file_name))
            if file_name[-5:] == ".rstx":
                file_paths_list_rstx.append((root, file_name))
            if file_name[-3:] == ".py":
                file_paths_list_py.append((root, file_name))

    return file_paths_list, file_paths_list_rstx, file_paths_list_py


file_paths_list, file_paths_list_rstx, file_paths_list_py = get_file_paths()

for root, file_name in file_paths_list_rstx:
    file_path = root + "/" + file_name

    with open(file_path, "r") as f:
        lines = f.readlines()

    for iline, line in enumerate(lines):
        if ".. embed-test" in line:
            lines[iline] = process_test(root, file_name, iline, line)
        elif ".. embed-options-table" in line:
            lines[iline] = process_options(root, file_name, iline, line)

    new_lines = []
    for line in lines:
        if isinstance(line, list):
            new_lines.extend(line)
        else:
            new_lines.append(line)

    new_file_path = file_path[:-5] + ".rst"
    with open(new_file_path, "w") as f:
        f.writelines(new_lines)
