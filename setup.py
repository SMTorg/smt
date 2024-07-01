"""
Author: John T. Hwang <hwangjt@umich.edu>
        Mohamed A. Bouhlel <mbouhlel@umich.edu>
        Remi Lafage <remi.lafage@onera.fr>
        Lucas Alber <lucasd.alber@gmail.com>
        Paul Saves <paul.saves@onera.fr>

This package is distributed under New BSD license.
"""

from setuptools import setup, Extension
import sys
import numpy as np
from Cython.Build import cythonize

from smt import __version__

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: C++
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.8
Programming Language :: Python :: Implementation :: CPython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: MacOS
"""

LONG_DESCRIPTION = """
The surrogate modeling toolbox (SMT) is a Python package that contains \
a collection of surrogate modeling methods, sampling techniques, and \
benchmarking functions. This package provides a library of surrogate \
models that is simple to use and facilitates the implementation of additional methods. \

SMT is different from existing surrogate modeling libraries because of \
its emphasis on derivatives, including training derivatives used for \
gradient-enhanced modeling, prediction derivatives, and derivatives \
with respect to the training data. It also includes new surrogate models \
that are not available elsewhere: kriging by partial-least squares reduction \
and energy-minimizing spline interpolation.

SMT 2.0 adds the capability to handle mixed-variable surrogate models \
and hierarchical variables.
"""

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


metadata = dict(
    name="smt",
    version=__version__,
    description="The Surrogate Modeling Toolbox (SMT)",
    long_description=LONG_DESCRIPTION,
    author="Remi Lafage et al.",
    author_email="remi.lafage@onera.fr",
    license="BSD-3",
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    packages=[
        "smt",
        "smt.surrogate_models",
        "smt.problems",
        "smt.sampling_methods",
        "smt.utils",
        "smt.applications",
    ],
    install_requires=[
        "scikit-learn",
        "pyDOE3",
        "scipy",
        "jenn",
    ],
    extras_require={
        "numba": [  # pip install smt[numba]
            "numba~=0.56.4",
        ],
        "cs": [  # pip install smt[cs]
            "ConfigSpace~=0.6.1",
        ],
        "gpx": ["egobox~=0.20"],  # pip install smt[gpx]
    },
    python_requires=">=3.8",
    zip_safe=False,
    ext_modules=ext,
    url="https://github.com/SMTorg/smt",  # use the URL to the github repo
    download_url="https://github.com/SMTorg/smt/releases",
)

setup(**metadata)
