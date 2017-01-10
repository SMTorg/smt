from numpy.distutils.core import setup, Extension
import os


if os.name == 'nt':
    # If OS is Windows, don't compile Fortran code
    ext = []
else:
    # If OS is OS X or Linux, assume Fortran compilers are available
    ext = Extension(name='smt.lib',
        extra_compile_args=['-fbounds-check'],
        sources=[
            'src_f/tps.f95',
            'src_f/rbf.f95',
            'src_f/idw.f95',
            'src_f/utils.f95',
        ],
    )

setup(name='smt',
    version='0.1',
    description='The Surrogate Model Toolbox (SMT)',
    author='Mohamed Amine Bouhlel',
    author_email='mbouhlel@umich.edu',
    license='BSD-3',
    packages=['smt'],
    install_requires=[
    'scikit-learn',
    'pyDOE'
    ],
    zip_safe=False,
    ext_modules=[ext],
)
