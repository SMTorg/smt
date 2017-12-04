'''
Author: Dr. John T. Hwang <hwangjt@umich.edu>
        Dr. Mohamed A. Bouhlel <mbouhlel@umich>

This package is distributed under New BSD license.
'''

from setuptools import setup, Extension
import os
import sys
from subprocess import call
import numpy as np

try:
    import Cython
except ImportError:
    import pip
    pip.main(['install', 'Cython'])

from Cython.Build import cythonize

extra_compile_args=[]
if not sys.platform.startswith('win'):
    extra_compile_args.append('-std=c++11')

ext = cythonize(
    Extension("smt.methods.rbfclib",
    sources=[
        'smt/src/rbf/rbf.cpp',
        'smt/src/rbf/rbfclib.pyx',
    ],
    language="c++", extra_compile_args=extra_compile_args,
    include_dirs=[np.get_include(),
])) + cythonize(
    Extension("smt.methods.idwclib",
    sources=[
        'smt/src/idw/idw.cpp',
        'smt/src/idw/idwclib.pyx',
    ],
    language="c++", extra_compile_args=extra_compile_args,
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
    language="c++", extra_compile_args=extra_compile_args,
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
        'matplotlib',
        'numpydoc',
        'six>=1.10',
        'scipy>=1.0.0'
    ],
    zip_safe=False,
    ext_modules=ext,
    url = 'https://github.com/SMTorg/smt', # use the URL to the github repo
    download_url = 'https://github.com/SMTorg/smt/archive/v0.1.tar.gz',
)
