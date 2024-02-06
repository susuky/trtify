


import tensorrt as trt

from trtify.builder import build_engine, create_profiles
from trtify.encryption import Cryptography
from trtify.utils import Null


__all__ = [
    'create_network',
    'parse_onnx',
    'BaseNetworkDefinition',
    'NetworkDefinition',
]

def create_network(
    implicit_batch=False,
    log_severity=trt.Logger.INFO,
    gLOGGER: trt.Logger = None,
) -> tuple[trt.Builder, trt.INetworkDefinition]:
    LOGGER = gLOGGER or trt.Logger(log_severity)
    builder = trt.Builder(LOGGER)
    flags = (
        0 if implicit_batch else 1 << (int)(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )
    )
    network = builder.create_network(flags)
    network = NetworkDefinition(network)
    network.builder = builder
    return builder, network


def parse_onnx(network,
               onnx_path='test.onnx',
               gLOGGER=None,
               debug=False,
               **kwargs):
    '''
    token (bytes): None
    keyfname (str): None
    '''
    LOGGER = gLOGGER or trt.Logger(kwargs.get('min_severity', trt.Logger.INFO))

    if isinstance(network, NetworkDefinition):
        network = network.network

    parser = trt.OnnxParser(network, LOGGER)

    with open(onnx_path, 'rb') as f:
        binary = f.read()
        token = kwargs.get('token', None)
        keyfname = kwargs.get('keyfname', None)
        if token or keyfname:
            cryptography = Cryptography(token=token, keyfname=keyfname)
            if not isinstance(cryptography, Null):
                binary = cryptography.decrypt(binary)

        success = parser.parse(binary)
        if debug and not success:
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None


class BaseNetworkDefinition:
    _builder = None

    def __init__(self, network: trt.INetworkDefinition):
        self.network = network
        self.profiles = None
        self.trtLOGGER = None

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, o):
        self._builder = o
        self.trtLOGGER = o.logger

    def set_optimization_profiles(self, *configs):
        self.profiles = create_profiles(self.builder, *configs)

    def del_optimization_profiles(self):
        self.profiles = None

    def build(self,
              engine_path='engine.🔥',
              profiles=None,
              workspace=1 << 30,
              fp16=False,
              int8=False,
              int8_calibrator=None,
              dla_core=None,
              gpu_fallback=True,
              with_encrypt=False,
              token=None,
              key_path='',
              verbose_token=False,
              **kwargs):
        '''
        See `build_engine`
        '''
        profiles = self.profiles or profiles
        return build_engine(builder=self.builder,
                            network=self.network,
                            engine_path=engine_path,
                            profiles=profiles,
                            workspace=workspace,
                            fp16=fp16,
                            int8=int8,
                            int8_calibrator=int8_calibrator,
                            dla_core=dla_core,
                            gpu_fallback=gpu_fallback,
                            with_encrypt=with_encrypt,
                            token=token,
                            key_path=key_path,
                            verbose_token=verbose_token,
                            **kwargs)

    def parse_onnx(self,
                   onnx_path='test.onnx',
                   token=None,
                   debug=False,
                   **kwargs):
        '''
        See `parse_onnx`
        '''
        parse_onnx(network=self.network,
                   onnx_path=onnx_path,
                   gLOGGER=self.trtLOGGER,
                   token=token,
                   debug=debug,
                   **kwargs)

    def __getattr__(self, name):
        '''
        If attr is not defined in self, it would get self.network attr
        '''
        if hasattr(self.network, name):
            return getattr(self.network, name)


def _get_layers():
    from . import layer_mixins
    
    layers =  (
        getattr(layer_mixins, d) 
        for d in dir(layer_mixins) 
        if d.endswith('_Mixin')
    )
    return layers


class NetworkDefinition(BaseNetworkDefinition, *_get_layers()): ...

