
import os
import shutil

from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages  

VERSION = '0.0.1'
CYTHONIZE = os.getenv('CYTHONIZE', '0')

if CYTHONIZE.lower() in ('1', 'true'):
    modules = [
        'builder',
        'cuda_backend',
        'encryption',
        'network_definition',
        'runtime',
        'utils',
        '__init__'
    ]

    for module in modules:
        shutil.copy(f'trtify/{module}.py', f'trtify/{module}.pyx')

    extensions = [
        Extension(f'trtify.{module}', [f'trtify/{module}.pyx'])
        for module in modules
    ]

    setup(
        name='trtify', 
        python_requires='>=3.8',
        ext_modules=cythonize(extensions), 
        version=VERSION,
    )
else:
    setup(
        name='trtify', 
        python_requires='>=3.8',
        version=VERSION,
        packages=find_packages(),
    )    

