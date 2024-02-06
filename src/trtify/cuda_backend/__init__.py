

import numpy as np
import os
import warnings

from .base import BaseCudaBackend
from ..utils import get_packages

__all__ = [
    'CUDA_BACKEND',
    'CudaAPI'
]

_implemented_backends = (
    'pycuda',
    'cuda-python',
)


def _get_backend() -> tuple[str, list[str]]:
    packages = get_packages()
    available = [c for c in _implemented_backends if c in packages]
    if not available:
        raise ImportError('No available CUDA backend')

    backend = os.getenv('CUDA_BACKEND', available[0])
    if backend not in available:
        warnings.warn(
            f'CUDA Backend {backend} is not available, '
            f'fallback to {available[0]}'
        )
        backend = available[0]
    return backend, available


CUDA_BACKEND, AVAILABLE_BACKEND = _get_backend()

_backend = {}
if 'pycuda' in AVAILABLE_BACKEND:
    from . import _pycuda
    _backend['pycuda'] = _pycuda
if 'cuda-python' in AVAILABLE_BACKEND:
    from . import _cuda_python
    _backend['cuda-python'] = _cuda_python

CudaAPI = _backend.get(CUDA_BACKEND).CudaAPI


def switch_backend(backend: str):
    global CUDA_BACKEND
    global CudaAPI

    if backend not in AVAILABLE_BACKEND:
        raise ValueError(f'Backend `{backend}` is not available')
    
    CUDA_BACKEND = backend
    CudaAPI = _backend.get(CUDA_BACKEND).CudaAPI
    