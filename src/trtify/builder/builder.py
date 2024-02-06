
import tensorrt as trt

from trtify.encryption import Cryptography
from trtify.utils import Null

__all__ = [
    'create_profile_config',
    'create_profiles',
    'build_engine',
]

VERSION = trt.__version__


def create_profile_config(binding, opt_shape, min_shape=None, max_shape=None):
    '''
    Example:
        >>> create_profile_config('input', (1, 3, 32, 32))

        >>> create_profile_config(['input1', 'input2'], (1, 3, 32, 32))

        >>> create_profile_config(
            binding=['input1', 'input2', 'input3'], 
            opt_shape=[(1, 3, 32, 32), (1, 42), (1, 64)],
            min_shape=[(1, 3, 32, 32), (1, 42), (1, 64)],
            max_shape=[(2, 3, 32, 66), (2, 50), (2, 80)],
        )

        >>> create_profile_config(
            binding=['input1', 'input2', 'input3'], 
            opt_shape=[(1, 3, 32, 32), (1, 42), (1, 64)],
            min_shape=[None, None, (1, 32)],
            max_shape=[None, (2, 50), None],
        )
    '''
    config = {}

    binding = binding if isinstance(binding, list) else [binding]

    if isinstance(opt_shape, list):
        if None in opt_shape:
            raise ValueError(
                f'Get opt_shape:{opt_shape}. `None` should not in opt_shape'
            )
        if len(opt_shape) != len(binding):
            raise ValueError(
                f'Get opt_shape:{opt_shape}.'
                f'Number of opt_shape and binding should be same.'
            )
    else:
        opt_shape = [opt_shape for _ in range(len(binding))]

    min_shape = min_shape or opt_shape
    if isinstance(min_shape, list):
        if len(min_shape) != len(binding):
            raise ValueError(
                f'Get min_shape:{min_shape}.'
                f'Number of min_shape and binding should be same.'
            )
    else:
        min_shape = [min_shape for _ in range(len(binding))]

    max_shape = max_shape or opt_shape
    if isinstance(max_shape, list):
        if len(max_shape) != len(binding):
            raise ValueError(
                f'Get max_shape:{max_shape}.'
                f'Number of max_shape and binding should be same.'
            )
    else:
        max_shape = [max_shape for _ in range(len(binding))]

    for b, opt, min, max in zip(binding, opt_shape, min_shape, max_shape):
        config[b] = {
            'opt_shape': opt,
            'min_shape': min,
            'max_shape': max,
        }
    return config


def create_profiles(builder, *configs):
    '''
    Example1:
        config = {'input1': {'shape': (1, 3, 224, 224)}
        profiles = create_profiles(builder, config)

    Example2:
        config1 = {'input1': {'shape': (1, 3, 224, 224)}, 'input2': {'shape': (1, 224)}}
        config2 = {
            'input1': {
                'opt_shape': (1, 3, 224, 224),
                'min_shape': (1, 3, 224, 224),
                'max_shape': (1, 3, 448, 448),
            },
            'input2': {
                'opt_shape': (1, 3, 48, 48),
            }
        }
        profiles = create_profiles(builder, config1, config2)

    for profile in profiles:
        config.add_optimization_profile(profile)
    '''
    profiles = []
    for config in configs:
        profile = builder.create_optimization_profile()
        for input, shapes in config.items():
            opt_shape = shapes.get('opt_shape', shapes.get('shape', None))
            if opt_shape is None:
                raise ValueError(
                    f'config should have either `shape` or `opt_shape`'
                )
            if not isinstance(opt_shape, tuple):
                raise TypeError(
                    f'The dtpye of shape inside config should be tuple'
                )

            min_shape = shapes.get('min_shape', opt_shape) or opt_shape
            max_shape = shapes.get('max_shape', opt_shape) or opt_shape
            profile.set_shape(input=input,
                              min=min_shape,
                              opt=opt_shape,
                              max=max_shape)
        profiles.append(profile)
    return profiles


def build_engine(
    builder: trt.Builder,
    network: trt.INetworkDefinition,
    engine_path: str = 'engine.🔥',
    profiles: trt.IOptimizationProfile = None,
    **kwargs
) -> None:
    '''
    Args:
        builder
        network
        engine_path
        profiles (list[trt.IOptimizationProfile]|trt.IOptimizationProfile)
    Kwargs:
        workspace (int): 1 GiB

        fp16 (bool): False
        int8 (bool): False
        int8_calibrator: None

        dla_core: None
        gpu_fallback: True

        with_encrypt (bool): False
        token (bytes): None
        key_path (str): None
        verbose_token (bool): True
    '''
    from ..network_definition import NetworkDefinition
    config = builder.create_builder_config()

    if isinstance(network, NetworkDefinition):
        network = network.network

    # implicit batch
    if network.has_implicit_batch_dimension:
        builder.max_batch_size = kwargs.pop('bs', 1)

    # working space
    workspace = kwargs.get('workspace', 1 << 30)
    if VERSION < '8.4':
        config.max_workspace_size = workspace
    else:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)

    # use fp16
    use_fp16 = kwargs.get('fp16', kwargs.get(
        'half', kwargs.get('use_fp16', False)))
    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # use int8
    if kwargs.get('int8', False) and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = kwargs.get('int8_calibrator', None)
        if not config.int8_calibrator:
            raise ValueError('To run int8, you need to set int8_calibrator')

    # setting DLA
    dla_core = kwargs.get('dla_core')
    if isinstance(dla_core, int) and builder.num_DLA_cores > dla_core >= 0:

        if not config.get_flag(trt.BuilderFlag.FP16) or config.get_flag(trt.BuilderFlag.INT8):
            raise ValueError(
                'DLA have to run with fp16 or int8.'
                'To enable it, use `fp16=True` or `int8=True`.'
            )
        config.DLA_core = dla_core
        config.default_device_type = trt.DeviceType.DLA

        # enable fallback to normal GPU in case DLA layers are not supported.
        if kwargs.get('gpu_fallback', True):
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    # add optimization profile for dynamic shape
    # To work with dynamic shape and shape input,
    # optimization profile should be defined.
    if profiles is not None:
        if isinstance(profiles, trt.IOptimizationProfile):
            profiles = (profiles, )

        for profile in profiles:
            config.add_optimization_profile(profile)
            config.set_calibration_profile(profile)

    if 1:  # build
        if VERSION < '8.4':
            binary = builder.build_engine(network, config).serialize()
        else:
            binary = builder.build_serialized_network(network, config)

        if kwargs.get('with_encrypt', False):
            cryptography = Cryptography(token=kwargs.get('token', None))
            if not isinstance(cryptography, Null):
                key_path = kwargs.get('key_path', '')
                if key_path:
                    cryptography.export_key(key_path)
                if kwargs.get('verbose_token'):
                    print(f'token: {cryptography.token}')
                binary = cryptography.encrypt(binary)

    # save binary file
    if engine_path is not None:
        with open(engine_path, 'wb') as f:
            f.write(binary)
    return binary
