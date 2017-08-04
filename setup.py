from numpy.distutils.core import setup, Extension
import os
from subprocess import call
from Cython.Build import cythonize
import numpy as np


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
    ],
    zip_safe=False,
    ext_modules=ext,
)
else:
    # If OS is OS X or Linux, assume Fortran compilers are available
    ext = [
        Extension(name='smt.methods.IDWlib', extra_compile_args=['-fbounds-check'],
            sources=['src_f/idw.f95']),
        Extension(name='smt.methods.RBFlib', extra_compile_args=['-fbounds-check'],
            sources=['src_f/rbf.f95']),
        Extension(name='smt.methods.RMTSlib', extra_compile_args=['-fbounds-check'],
            sources=['src_f/rmts.f95', 'src_f/utils.f95']),
        Extension(name='smt.methods.RMTClib', extra_compile_args=['-fbounds-check'],
            sources=['src_f/rmtc.f95', 'src_f/utils.f95']),
        Extension(name='smt.methods.RMTBlib', extra_compile_args=['-fbounds-check'],
            sources=['src_f/rmtb.f95', 'src_f/utils.f95']),
    ] + cythonize(
        Extension("smt.methods.rbfclib",
        sources=[
            'smt/src/rbf.cpp',
            'smt/src/rbfclib.pyx',
        ],
        language="c++", extra_compile_args=['-std=c++11'],
        include_dirs=[np.get_include()]
    ))
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
    ],
    zip_safe=False,
    ext_modules=ext,
)



for lib_name in ['IDWlib', 'RBFlib', 'RMTSlib', 'RMTClib', 'RMTBlib']:
    try:
        call(['mv', 'smt/%s.cpython-36m-darwin.so' % lib_name, 'smt/%s.so' % lib_name])
    except:
        pass
