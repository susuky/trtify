
import os

from setuptools import setup, Extension, find_packages  

VERSION = '0.0.1'
CYTHONIZE = os.getenv('CYTHONIZE', '0')

if CYTHONIZE.lower() in ('1', 'true'):
    import atexit
    import shutil
    from Cython.Build import cythonize
    from pathlib import Path


    def clean(root='src', suffixes=('.pyx', '.c')):
        for suffix in suffixes:
            for path in Path(root).rglob(f'*{suffix}'):
                path.unlink()
    atexit.register(clean)


    module_paths = [
        *Path('src').rglob('*.py'),
    ]

    extensions = []
    for path in module_paths:
        path_pyx = path.with_suffix('.pyx')
        shutil.copy(path, path_pyx)

        module = '.'.join((*path.parts[1:-1], path.stem))
        extensions.append(Extension(module, [path_pyx.as_posix()]))

    setup(
        name='trtify', 
        python_requires='>=3.8',
        ext_modules=cythonize(extensions), 
        version=f'{VERSION}+cythonized',
    )

else:
    setup(
        name='trtify', 
        python_requires='>=3.8',
        version=VERSION,
        package_dir={'': 'src'},
        packages=find_packages(where='src', include=['trtify', 'trtify.*']),
    )    


