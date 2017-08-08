from numpy.distutils.core import setup, Extension
import os
from subprocess import call
import numpy as np
from Cython.Build import cythonize


if os.name == 'nt':
    # If OS is Windows, don't compile Fortran code
    ext = []
    setup(name='smt',
    version='0.1',
    description='The Surrogate Model Toolbox (SMT)',
    author='Mohamed Amine Bouhlel',
    author_email='mbouhlel@umich.edu',
    license='BSD-3',
    packages=[
        'smt',
        'smt/methods',
        'smt/problems',
        'smt/sampling',
        'smt/utils',
    ],
    install_requires=[
        'scikit-learn',
        'pyDOE',
        'matplotlib',
        'numpydoc',
    ],
    zip_safe=False,
    ext_modules=ext,
)
else:
    # If OS is OS X or Linux, assume Fortran compilers are available
    ext = cythonize(
        Extension("smt.methods.rbfclib",
        sources=[
            'smt/src/rbf/rbf.cpp',
            'smt/src/rbf/rbfclib.pyx',
        ],
        language="c++", extra_compile_args=['-std=c++11'],
        include_dirs=[np.get_include(),
    ])) + cythonize(
        Extension("smt.methods.idwclib",
        sources=[
            'smt/src/idw/idw.cpp',
            'smt/src/idw/idwclib.pyx',
        ],
        language="c++", extra_compile_args=['-std=c++11'],
        include_dirs=[np.get_include(),
    ])) + cythonize(
        Extension("smt.methods.rmtsclib",
        sources=[
            'smt/src/rmts/rmtsclib.pyx',
            'smt/src/rmts/utils.cpp',
            'smt/src/rmts/rmts.cpp',
            'smt/src/rmts/rmtb.cpp',
            'smt/src/rmts/rmtc.cpp',
        ],
        language="c++", extra_compile_args=['-std=c++11'],
        include_dirs=[np.get_include(),
    ]))
    setup(name='smt',
    version='0.1',
    description='The Surrogate Model Toolbox (SMT)',
    author='Mohamed Amine Bouhlel',
    author_email='mbouhlel@umich.edu',
    license='BSD-3',
    packages=[
        'smt',
        'smt/methods',
        'smt/problems',
        'smt/sampling',
        'smt/utils',
    ],
    install_requires=[
        'scikit-learn',
        'pyDOE',
        'pyamg',
        'matplotlib',
        'numpydoc',
    ],
    zip_safe=False,
    ext_modules=ext,
)
