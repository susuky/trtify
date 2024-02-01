
'''
Note: 
    Because the subclass of BaseCudaBackend cannot be instantiated.
    Therefore, the abc module and abstract methods just tell which methods are needed.

'''
import abc
import numpy as np


class NoInstance(type):
    def __call__(self, *args, **kwargs):
        raise TypeError('Cannot instantiate!')
    

class _Base(abc.ABCMeta, NoInstance): ...


class BaseCudaBackend(_Base):
    BACKEND = 'base'
    Stream = None

    @abc.abstractmethod
    def pagelocked_empty(size: int, dtype: np.dtype) -> np.ndarray:
        pass

    @abc.abstractmethod
    def mem_alloc(nbytes: int) -> int:
        pass

    @abc.abstractmethod
    def memcpy_htod(device: int, host: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def memcpy_dtoh(host: np.ndarray, device: int) -> None:
        pass

    @abc.abstractmethod
    def create_stream() -> int:
        pass

    @abc.abstractmethod
    def get_stream_handle(stream) -> int:
        pass

    @abc.abstractmethod
    def stream_destroy(stream) -> None:
        pass

    @abc.abstractmethod
    def memcpy_htod_async(device, host, stream) -> None:
        pass

    @abc.abstractmethod
    def memcpy_dtoh_async(host, device, stream) -> None:
        pass

    @abc.abstractmethod
    def synchronize(stream) -> None:
        pass

    @abc.abstractmethod
    def free_host(host) -> None:
        pass

    @abc.abstractmethod
    def free_device(device) -> None:
        pass
