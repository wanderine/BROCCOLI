#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib

setup(name='BROCCOLI',
      version='1.0',
      description='An open source multi-platform software for parallel analysis of fMRI data on many-core CPUs and GPUs',
      author='Anders Eklund',
      author_email='andek034@gmail.com',
      url='https://github.com/wanderine/BROCCOLI',
      packages=['broccoli', 'nipype.interfaces.broccoli'],
      data_files=[(get_python_lib() + '/broccoli', ['broccoli/_broccoli_base.so', '../BROCCOLI_LIB/broccoli_lib_kernel.cpp'])],
      scripts=['RegisterEPIT1.py', 'RegisterT1MNI.py']
     )
