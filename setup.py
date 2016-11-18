import os, sys
from setuptools import setup
from distutils.core import Extension
import subprocess
import shutil
 

try:
    subprocess.call(['make'])
except:
    file = open(os.getcwd()+"\__init__.py",'w')
    file.write("from smt.kpls import KPLS\n")
    file.write("from smt.ls import LS\n")   
    file.write("from smt.pa2 import PA2")
    file.close()

setup(name='smt',
      version='0.1',
      description='The Surrogate Model Toolbox (SMT)',
      author='Mohamed Amine Bouhlel',
      author_email='mbouhlel@umich.edu',
      license='BSD-3',
      packages=['smt'],
      install_requires=['scikit-learn'],
      zip_safe=False)

try:
    myFile = os.getcwd()+"/smt/lib.so"
    version = sys.version_info
    shutil.copy(myFile,"/usr/local/lib/python"+str(version[0])+"."+str(version[1])+
            "/dist-packages/smt-0.1-py"+str(version[0])+"."+str(version[1])+
            ".egg/smt/lib.so")
except:
    pass
