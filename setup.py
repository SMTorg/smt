"""Minimal setup.py for building Cython/C++ extensions.

All project metadata is in pyproject.toml.
"""

import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extra_compile_args = []
if not sys.platform.startswith("win"):
    extra_compile_args.append("-std=c++11")

ext = (
    cythonize(
        Extension(
            "smt.surrogate_models.rbfclib",
            sources=["smt/src/rbf/rbf.cpp", "smt/src/rbf/rbfclib.pyx"],
            language="c++",
            extra_compile_args=extra_compile_args,
            include_dirs=[np.get_include()],
        )
    )
    + cythonize(
        Extension(
            "smt.surrogate_models.idwclib",
            sources=["smt/src/idw/idw.cpp", "smt/src/idw/idwclib.pyx"],
            language="c++",
            extra_compile_args=extra_compile_args,
            include_dirs=[np.get_include()],
        )
    )
    + cythonize(
        Extension(
            "smt.surrogate_models.rmtsclib",
            sources=[
                "smt/src/rmts/rmtsclib.pyx",
                "smt/src/rmts/utils.cpp",
                "smt/src/rmts/rmts.cpp",
                "smt/src/rmts/rmtb.cpp",
                "smt/src/rmts/rmtc.cpp",
            ],
            language="c++",
            extra_compile_args=extra_compile_args,
            include_dirs=[np.get_include()],
        )
    )
)

setup(ext_modules=ext)
