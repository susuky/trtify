
import ctypes
import numpy as np

from cuda import cudart
from cuda import cuda

from .base import BaseCudaBackend


__all__ = [
    'check_cuda_err',
    'cuda_call',
    'CudaAPI',
]


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


class CudaAPI(metaclass=BaseCudaBackend):
    BACKEND = 'cuda-python'
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
