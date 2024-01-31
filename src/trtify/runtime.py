
import numpy as np
import tensorrt as trt

from .cuda_backend import CudaAPI as cuda
from .encryption import Cryptography
from .utils import check_dim, Null, Timer

__all__ = [
    'cuda',
    'TRTRuntime',
    'load_engine',
    'HostDeviceMem',
    'allocate_buffers',
    'memcpy_host_to_device',
    'memcpy_device_to_host',
    'do_inference',
    'do_inference_v2',
]


TRT_VERSION = trt.__version__


class HostDeviceMem:
    def __init__(self, size: int, dtype: np.dtype, **kwargs):
        """
        Initializes a new instance of the class.
        
        Args:
            size (int): The size of the array.
            dtype (np.dtype): The data type of the array.
            **kwargs: Additional keyword arguments.
        
        Attributes:
            casting (str): The casting method to be used.
            size (int): The size of the array.
            dtype (np.dtype): The data type of the array.
            _nbytes (int): The total number of bytes in the array.
            _host (np.ndarray): The host memory for the array.
            _device (int): The device memory for the array.
        """
        self.casting = kwargs.get('casting', 'no')

        self.size = size
        self.dtype = dtype
        self._nbytes = size * dtype.itemsize

        self._host = cuda.pagelocked_empty(size, dtype)
        self._device = cuda.mem_alloc(self.nbytes)

    @property
    def nbytes(self):
        return self._nbytes

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f'Tried to fit an array of size {arr.size} into host memory of size {self.host.size}'
            )
        np.copyto(self.host[:arr.size], arr.ravel(), casting=self.casting)

    @property
    def device(self):
        return self._device

    def __repr__(self):
        return f"Host:\n\t{self.host}\nDevice:\n\t{self.device}\nSize:\n\t{self.nbytes}\n"

    def free(self):
        cuda.free_host(self.host)
        cuda.free_device(self.device)


class TRTRuntime:
    def __init__(self, engine_path='engine.🔥',
                 shape: [tuple, dict[str, tuple]] = None,
                 dtype=np.float32,
                 log_severity: str = 'INFO',
                 gLOGGER: trt.Logger = None,
                 token: bytes = None,
                 keyfname: str = None,
                 ) -> None:
        '''
        log_severity (str): 
            available: {'INTERNAL_ERROR', 'ERROR', 'WARNING', 'INFO', 'VERBOSE'}
        '''
        if not isinstance(log_severity, trt.ILogger.Severity):
            log_severity = getattr(trt.Logger, log_severity, trt.Logger.INFO)
        LOGGER = gLOGGER or trt.Logger(log_severity)

        context, engine = load_engine(engine_path,
                                      gLOGGER=LOGGER,
                                      token=token,
                                      keyfname=keyfname)

        inps, outs, bindings, stream = allocate_buffers(context, shape)
        self.__dict__.update(locals())
        self.warmup()

    def warmup(self, times=10):
        stream = self.stream
        context = self.context
        inps = self.inps
        outs = self.outs
        bindings = self.bindings
        timer = Timer()

        for i in range(times):
            for mem in inps:
                mem.host = np.random.random(*mem.host.shape).astype(self.dtype)

            with timer:
                out = self.do_inference_v2(context,
                                           bindings,
                                           inps,
                                           outs,
                                           stream)
            print(f'warmup -> {timer.dt * 1E3:.1f}ms')

    def do_infer(self, inps: list[HostDeviceMem]):
        return do_inference_v2(self.context,
                               self.bindings,
                               inps,
                               self.outs,
                               self.stream)

    def do_inference_v2(self, context, bindings, inps, outs, stream):
        return do_inference_v2(context, bindings, inps, outs, stream)


def load_engine(
    engine_path='engine.🔥',
    log_severity=trt.Logger.INFO,
    gLOGGER: trt.Logger = None,
    **kwargs
) -> tuple[trt.IExecutionContext, trt.ICudaEngine]:
    '''
    Kwargs:
        token (bytes) : None
        keyfname (str): None
    '''
    LOGGER = gLOGGER or trt.Logger(log_severity)
    runtime = trt.Runtime(LOGGER)
    with open(engine_path, 'rb') as f:
        engine = f.read()

        token = kwargs.get('token', None)
        keyfname = kwargs.get('keyfname', None)
        if token or keyfname:
            cryptography = Cryptography(token=token, keyfname=keyfname)
            if not isinstance(cryptography, Null):
                engine = cryptography.decrypt(engine)
        engine = runtime.deserialize_cuda_engine(engine)

    engine = engine or Null()
    context = engine.create_execution_context()
    return context, engine


def isinput(binding, engine):
    if binding not in engine:
        return False
    if TRT_VERSION < '8.5':
        return engine.binding_is_input(binding)
    return engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT


def setup_profile(context, shape=None, stream=None):
    '''
    shape (dict[str, tuple]| tuple)
    '''

    engine = context.engine
    num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles

    input_tensors = [
        (binding_index, binding)
        for (binding_index, binding) in enumerate(engine)
        if 'profile' not in binding and isinput(binding, engine)
    ]

    if shape is None:
        for binding_idx, binding in input_tensors:
            if TRT_VERSION < '8.5':
                _, shape, _ = engine.get_profile_shape(0, binding_idx)
                context.set_binding_shape(binding_idx, shape)
            else:
                _, shape, _ = engine.get_tensor_profile_shape(binding, 0)
                context.set_input_shape(binding, shape)

    else:
        for profile_idx in range(engine.num_optimization_profiles):
            for binding_idx, binding in input_tensors:
                binding_idx += num_binding_per_profile * profile_idx
                min_shape, opt_shape, max_shape = (
                    engine.get_profile_shape(profile_idx, binding_idx)
                    if TRT_VERSION < '8.5' else
                    engine.get_tensor_profile_shape(binding, profile_idx)
                )

                s = (
                    shape.get(binding, ()) if isinstance(shape, dict) else
                    shape
                )
                if not (
                    check_dim(min_shape, s, '<=') and
                    check_dim(s, max_shape, '<=')
                ):
                    break
            else:
                # XXX: Change profile would cause memory issue
                if TRT_VERSION < '8.0' or stream is None:
                    context.active_optimization_profile = profile_idx
                else:
                    context.set_optimization_profile_async(profile_idx,
                                                           cuda.get_stream_handle(stream))
                break
        else:
            raise RuntimeError(
                'Could not find any profile that can run the shape')

        for binding_index, binding in input_tensors:
            binding_index += num_binding_per_profile * context.active_optimization_profile
            s = shape.get(binding, ()) if isinstance(shape, dict) else shape
            if TRT_VERSION < '8.5':
                context.set_binding_shape(binding_idx, s)
            else:
                context.set_input_shape(binding, s)
                pass


def allocate_buffers(
    context,
    shape=None,
    **kwargs
) -> tuple[list[HostDeviceMem], list[HostDeviceMem], list[int], cuda.Stream]:
    '''
    Note: Swich profile would get `Segmentation fault`
    '''
    inps = []
    outs = []
    bindings = []
    stream = cuda.create_stream()
    setup_profile(context, shape=shape, stream=stream)
    engine = context.engine
    num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles

    for binding_index, binding in enumerate(engine, context.active_optimization_profile * num_binding_per_profile):
        if 'profile' in binding:
            break

        is_input = (
            engine.binding_is_input(binding) if TRT_VERSION < '8.5' else
            engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT
        )

        shape = (
            context.get_binding_shape(binding_index) if TRT_VERSION < '8.5' else
            context.get_tensor_shape(binding)
        )

        size = trt.volume(shape)

        if engine.has_implicit_batch_dimension:
            size *= engine.max_batch_size

        dtype = np.dtype(
            trt.nptype(
                engine.get_binding_dtype(binding) if TRT_VERSION < '8.5' else
                engine.get_tensor_dtype(binding)
            )
        )

        # Allocate host and device buffers
        binding_memory = HostDeviceMem(size, dtype)

        # Append the device buffer to device bindings.
        bindings.append(int(binding_memory.device))

        # Append to the appropriate list.
        if is_input:
            inps.append(binding_memory)
        else:
            outs.append(binding_memory)
    return inps, outs, bindings, stream


def memcpy_host_to_device(mems: list[HostDeviceMem], stream):
    # Wrapper for cudaMemcpy which infers copy size and does error checking
    [
        cuda.memcpy_htod_async(mem.device, mem.host, stream)
        for mem in mems
    ]


def memcpy_device_to_host(mems: list[HostDeviceMem], stream):
    # Wrapper for cudaMemcpy which infers copy size and does error checking
    [
        cuda.memcpy_htod_async(mem.device, mem.host, stream)
        for mem in mems
    ]


def _do_inference_base(inps, outs, stream, execute_async):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inps]

    # Run inference.
    execute_async()

    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outs]

    # Synchronize the stream
    cuda.synchronize(stream)

    # Return only the host outs.
    return [out.host for out in outs]


def do_inference(context, bindings, inps, outs, stream, bs=1):
    '''
    for implicit batch mode
    inps and outs are expected to be lists of HostDeviceMem objects.
    '''
    def execute_async():
        context.execute_async(bs,
                              bindings=bindings,
                              stream_handle=cuda.get_stream_handle(stream))
    return _do_inference_base(inps, outs, stream, execute_async)


def do_inference_v2(context, bindings, inps, outs, stream):
    '''
    for explicit batch mode
    inps and outs are expected to be lists of HostDeviceMem objects.    
    '''
    def execute_async():
        context.execute_async_v2(bindings=bindings,
                                 stream_handle=cuda.get_stream_handle(stream))
    return _do_inference_base(inps, outs, stream, execute_async)
