

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from .base import BaseCudaBackend


__all__ = [
    'CudaAPI',
]

class CudaAPI(metaclass=BaseCudaBackend):
    BACKEND = 'pycuda'
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
    