
import numpy as np
import pkg_resources
import os
import warnings

from .utils import Null

__all__ = [
    'CUDA_BACKEND',
    'CudaAPI'
]


def _get_backend():
    packages = [pkg.key for pkg in pkg_resources.working_set]
    available = [c for c in ('pycuda', 'cuda-python') if c in packages]
    if not available:
        raise ImportError('No available CUDA backend')

    CUDA_BACKEND = os.getenv('CUDA_BACKEND', available[0])
    if CUDA_BACKEND not in available:
        warnings.warn(
            f'CUDA Backend {CUDA_BACKEND} is not available, '
            f'fallback to {available[0]}'
        )
        CUDA_BACKEND = available[0]
    return CUDA_BACKEND, available


CUDA_BACKEND, AVAILABLE_BACKEND = _get_backend()


class NoInstance(type):
    def __call__(self, *args, **kwargs):
        raise TypeError('Cannot instantiate!')


if CUDA_BACKEND == 'pycuda':

    import pycuda.autoinit
    import pycuda.driver as cuda

    class CudaAPI(metaclass=NoInstance):
        BACKEND = CUDA_BACKEND
        Stream = cuda.Stream

        def pagelocked_empty(size: int, dtype: np.dtype) -> np.ndarray:
            return cuda.pagelocked_empty(size, dtype)

        def mem_alloc(nbytes: int) -> cuda.DeviceAllocation:
            return cuda.mem_alloc(nbytes)

        def memcpy_htod(device, host) -> None:
            cuda.memcpy_htod(device, host)

        def memcpy_dtoh(host, device) -> None:
            cuda.memcpy_dtoh(host, device)

        def create_stream() -> cuda.Stream:
            return cuda.Stream()

        def get_stream_handle(stream):
            return stream.handle

        def stream_destroy(stream):
            # Note: It would not release properly
            del stream

        def memcpy_htod_async(device, host, stream) -> None:
            cuda.memcpy_htod_async(device, host, stream)

        def memcpy_dtoh_async(host, device, stream) -> None:
            cuda.memcpy_dtoh_async(host, device, stream)

        def synchronize(stream) -> None:
            stream.synchronize()

        def free_host(host) -> None:
            # Note: It would not release properly
            del host

        def free_device(device) -> None:
            device.free()


elif CUDA_BACKEND == 'cuda-python':
    import ctypes
    from cuda import cudart
    from cuda import cuda

    def check_cuda_err(err):
        if isinstance(err, cuda.CUresult):
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f'Cuda Error: {err}')
        if isinstance(err, cudart.cudaError_t):
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f'Cuda Runtime Error: {err}')
        else:
            raise RuntimeError(f'Unknown error type: {err}')

    def cuda_call(call):
        err, *res = call
        check_cuda_err(err)
        if len(res) == 1:
            res = res[0]
        return res

    class CudaAPI(metaclass=NoInstance):
        BACKEND = CUDA_BACKEND
        Stream = cudart.cudaStream_t

        def pagelocked_empty(size: int, dtype: np.dtype) -> np.ndarray:
            nbytes = size * dtype.itemsize
            host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
            pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
            return np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type),
                                         (size, ))

        def mem_alloc(nbytes: int) -> int:
            return cuda_call(cudart.cudaMalloc(nbytes))

        def memcpy_htod(device: int, host: np.ndarray) -> None:
            cuda_call(
                cudart.cudaMemcpy(device, host, host.nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            )

        def memcpy_dtoh(host: np.ndarray, device: int) -> None:
            cuda_call(
                cudart.cudaMemcpy(host, device, host.nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            )

        def create_stream() -> cudart.cudaStream_t:
            return cuda_call(cudart.cudaStreamCreate())

        def get_stream_handle(stream):
            return stream

        def stream_destroy(stream):
            cuda_call(cudart.cudaStreamDestroy(stream))

        def memcpy_htod_async(device, host, stream):
            cuda_call(
                cudart.cudaMemcpyAsync(device, host, host.nbytes,
                                       cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                                       stream)
            )

        def memcpy_dtoh_async(host, device, stream):
            cuda_call(
                cudart.cudaMemcpyAsync(host, device, host.nbytes,
                                       cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                                       stream)
            )

        def synchronize(stream) -> None:
            cuda_call(cudart.cudaStreamSynchronize(stream))

        def free_host(host) -> None:
            cuda_call(cudart.cudaFreeHost(host.ctypes.data))

        def free_device(device) -> None:
            cuda_call(cudart.cudaFree(device))
else:
    raise ImportError('No available CUDA backend')
