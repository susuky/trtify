
import numpy as np
import tensorrt as trt

from .builder import build_engine, create_profiles
from .encryption import Cryptography
from .utils import bitmask2int, broadcast_to, dim2axes, totuple, Null

__all__ = [
    'create_network',
    'parse_onnx',
    'NetworkDefinition',
]

VERSION = trt.__version__


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


class _NetworkDefinition:
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


class NetworkDefinition(_NetworkDefinition):

    def add_input(self, shape, name='input', dtype=trt.float32) -> trt.ITensor:
        '''
        Add an input tensor to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the input tensor will be added.
            shape (tuple): The shape of the input tensor.
            name (str, optional): The name of the input tensor.
            dtype (trt.DataType, optional): The data type of the input tensor.

        Returns:
            trt.ITensor: The input tensor added to the network.

        Examples:
            To add an input tensor with shape (3, 224, 224) and name "input":
            >>> input_tensor = add_input(network, shape=(3, 224, 224), name='input', dtype=trt.float32)
        '''
        inp = self.network.add_input(name=name,
                                     dtype=dtype,
                                     shape=shape)
        return inp

    def mark_output(self, layer, name='output') -> None:
        '''
        Mark a layer or tensor as an output of the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the layer or tensor belongs.
            layer (trt.ILayer or trt.ITensor): The layer or tensor to mark as an output.
            name (str, optional): The name to assign to the output tensor.

        Returns:
            None

        Examples:
            To mark a layer as an output with the name "output":
            >>> mark_output(network, layer, name='output')

            To mark an input tensor as an output with the default name "output":
            >>> mark_output(network, input_tensor)
        '''

        if isinstance(layer, trt.ILayer):
            layer = layer.get_output(0)

        layer.name = name
        self.network.mark_output(tensor=layer)

    def add_constant(self,
                     scale=1,
                     weights=None,
                     layer_name='',
                     shape=None,
                     dtype=np.float32,
                     debug=False) -> trt.ITensor:
        '''
        Add a constant tensor to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the constant tensor will be added.
            scale (float, optional): The scaling factor for the constant tensor.
            weights (numpy.ndarray, optional): The weights (data) for the constant tensor.
            layer_name (str, optional): The name of the constant layer.
            shape (tuple, optional): The shape of the constant tensor. If not provided, it is inferred from the weights.
            dtype (numpy.dtype, optional): The data type of the constant tensor.
            debug (bool, optional): If True, print the shape of the constant tensor.

        Returns:
            trt.ITensor: The constant ITensor added to the network.

        Examples:
            To add a constant tensor with scale 255 and shape matching an input tensor:
            >>> trt255 = add_constant(network, scale=255, shape=input.shape)

            To add a constant tensor with scale 255 and shape matching an output tensor:
            >>> trt255 = add_constant(network, scale=255, shape=output.get_output(0).shape, debug=True)
        '''
        if weights is not None:
            if layer_name:
                weights = weights[layer_name]
            shape = shape or weights.shape
        else:
            shape = shape or (1,)
            weights = np.ones(shape)

        weights *= scale
        weights = broadcast_to(weights, shape)
        weights = weights.astype(dtype)
        c = self.network.add_constant(shape=weights.shape,
                                      weights=weights).get_output(0)

        if debug:
            print(f'{layer_name or "constant"}: Output shape: {c.shape}')
        return c

    def add_elementwise(self,
                        inp0,
                        inp1,
                        op=trt.ElementWiseOperation.SUM,
                        dtype=np.float32,
                        debug=False) -> trt.ITensor:
        '''   
        Add an element-wise operation layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the element-wise layer will be added.
            inp0 (trt.ITensor or numpy.ndarray): The first input tensor or a constant value.
            inp1 (trt.ITensor or numpy.ndarray): The second input tensor or a constant value.
            op (trt.ElementWiseOperation): The element-wise operation type (
                                                    'SUM', 'SUB', 'PROD', 'DIV', 'POW', 'FLOOR_DIV',
                                                    'MAX', 'MIN', 
                                                    'AND', 'OR', 'XOR', 
                                                    'EQUAL', 'GREATER', 'LESS',
                                            ).
            dtype (numpy.dtype, optional): The data type of the constant value if used.
            debug (bool, optional): If True, print the shape of the resulting tensor.

        Returns:
            trt.ITensor: The element-wise ITensor added to the network.

        Examples:
            To add an element-wise SUM operation between two input tensors:
            >>> inp0_tensor = input0_tensor
            >>> inp1_tensor = input1_tensor
            >>> sum_layer = add_elementwise(network, inp0_tensor, inp1_tensor, op=trt.ElementWiseOperation.SUM)

            To add an element-wise SUB operation between an input tensor and a constant value:
            >>> inp0_tensor = input_tensor
            >>> sub_layer = add_elementwise(network, inp0_tensor, 5, op=trt.ElementWiseOperation.SUB, dtype=np.float32)
        '''
        if isinstance(inp0, trt.ITensor):
            shape = inp0.shape
            if not isinstance(inp1, trt.ITensor):
                inp1 = self.add_constant(weights=inp1,
                                         shape=shape,
                                         dtype=dtype,
                                         debug=debug)
        elif isinstance(inp1, trt.ITensor):
            shape = inp1.shape
            inp0 = self.add_constant(weights=inp0,
                                     shape=shape,
                                     dtype=dtype,
                                     debug=debug)
        else:
            if debug:
                raise TypeError(
                    f'One of the inputs, either inp0 or inp1, must be of type trt.ITensor.'
                )
            return

        value = self.network.add_elementwise(inp0, inp1, op).get_output(0)
        if debug:
            print(f'elementwise compute {op.name}: {value.shape}')
        return value

    def add_normalize(self,
                      input,
                      zscore=True,
                      mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225),
                      value=255,
                      debug=False) -> trt.ITensor:
        '''
        Add a normalization operation layer to the TensorRT network.
            1. divide `input` by `value`
            2. if zscore: do z-noramlize with `mean`, `std`

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the normalization layer will be added.
            input (trt.ITensor): The input tensor to be normalized.
            zscore (bool, optional): If True, perform Z-score normalization.
            mean (tuple, optional): The mean values for Z-score normalization.
            std (tuple, optional): The standard deviation values for Z-score normalization.
            value (int, optional): The value to divide the input tensor by.
            debug (bool, optional): If True, print the shape of the output tensor.

        Returns:
            trt.ITensor: The normalization ITensor added to the network.

        Examples:
            To add a normalization layer that divides the input tensor by 255:
            >>> input_tensor = input_tensor
            >>> normalization_layer = add_normalize(network, input_tensor, zscore=False, value=255)

            To add a Z-score normalization layer to the input tensor:
            >>> input_tensor = input_tensor
            >>> normalization_layer = add_normalize(network, input_tensor, zscore=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), debug=True)
        '''
        normalize = self.add_elementwise(input,
                                         value,
                                         op=trt.ElementWiseOperation.DIV,
                                         debug=debug)
        if zscore:
            normalize = self.add_elementwise(normalize,
                                             mean,
                                             op=trt.ElementWiseOperation.SUB,
                                             debug=debug)
            normalize = self.add_elementwise(normalize,
                                             std,
                                             op=trt.ElementWiseOperation.DIV,
                                             debug=debug)
        if debug:
            print('normalize layer: ', normalize.shape)
        return normalize

    def add_conv2d(self,
                   input,
                   weights,
                   layer_name='',
                   stride=1,
                   padding=0,
                   dilation=1,
                   groups=1,
                   debug=False) -> trt.ITensor:
        '''
        Add a 2D convolution layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the convolution layer will be added.
            input (trt.ITensor): The input tensor to be convolved.
            weights (dict): A dictionary containing layer weights (e.g., 'weight', 'bias') for convolution.
            layer_name (str, optional): The name of the layer.
            stride (int or tuple, optional): The convolution stride (e.g., 1 or (2, 2)).
            padding (int or tuple, optional): The padding to be applied before convolution.
            dilation (int or tuple, optional): The dilation factor for convolution.
            groups (int, optional): The number of groups for grouped convolution.
            debug (bool, optional): If True, print the shape of the output tensor.

        Returns:
            trt.ITensor: The convolution ITensor added to the network.

        Examples:
            To add a 2D convolution layer to an input tensor:
            >>> input_tensor = input_tensor
            >>> layer_weights = {'weight': weight_tensor, 'bias': bias_tensor}
            >>> conv_layer = add_conv2d(network, input_tensor, layer_weights, layer_name='conv1', stride=1, padding=1)

            To add a grouped convolution layer to an input tensor with custom stride and dilation:
            >>> input_tensor = input_tensor
            >>> layer_weights = {'weight': weight_tensor, 'bias': bias_tensor}
            >>> conv_layer = add_conv2d(network, input_tensor, layer_weights, layer_name='conv2', stride=(2, 2), dilation=2, groups=2, debug=True)
        '''
        name = layer_name
        layer_name = f'{layer_name}.' if layer_name else ''
        w = weights[f'{layer_name}weight']
        bias_name = f'{layer_name}bias'
        b = weights[bias_name] if bias_name in weights else trt.Weights()

        ks = tuple(w.shape[-2:])
        out_size = w.shape[0]

        layer = self.network.add_convolution_nd(input=input,
                                                num_output_maps=out_size,
                                                kernel_shape=ks,
                                                kernel=w,
                                                bias=b)
        layer.stride_nd = totuple(stride)
        layer.padding_nd = totuple(padding)
        layer.dilation_nd = totuple(dilation)
        layer.num_groups = groups

        layer = layer.get_output(0)
        if debug:
            print(f'{name}: {layer.shape}')
        return layer

    def add_bn2d(self,
                 input,
                 weights,
                 layer_name='',
                 eps=1e-5,
                 debug=False) -> trt.ITensor:
        '''
        Add a 2D batch normalization layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the batch normalization layer will be added.
            input (trt.ITensor): The input tensor to be batch normalized.
            weights (dict): A dictionary containing layer weights (e.g., 'weight', 'bias', 'running_mean', 'running_var') for batch normalization.
            layer_name (str, optional): The name of the layer.
            eps (float, optional): The epsilon value for numerical stability.
            debug (bool, optional): If True, print the shape of the output tensor.

        Returns:
            trt.ITensor: The batch normalization ITensor added to the network.

        Examples:
            To add a 2D batch normalization layer to an input tensor:
            >>> input_tensor = input_tensor
            >>> layer_weights = {'weight': gamma_tensor, 'bias': beta_tensor, 'running_mean': mean_tensor, 'running_var': var_tensor}
            >>> bn_layer = add_bn2d(network, input_tensor, layer_weights, layer_name='bn1', eps=1e-5)

            To add a batch normalization layer to an input tensor with a custom epsilon value:
            >>> input_tensor = input_tensor
            >>> layer_weights = {'weight': gamma_tensor, 'bias': beta_tensor, 'running_mean': mean_tensor, 'running_var': var_tensor}
            >>> bn_layer = add_bn2d(network, input_tensor, layer_weights, layer_name='bn2', eps=1e-3, debug=True)
        '''
        layer_name = f'{layer_name}.' if layer_name else ''
        gamma = weights[f'{layer_name}weight']
        beta = weights[f'{layer_name}bias']
        mean = weights[f'{layer_name}running_mean']
        var = weights[f'{layer_name}running_var']
        var = np.sqrt(var + eps)

        scale = gamma / var
        shift = -mean / var * gamma + beta
        bn = self.network.add_scale_nd(input=input,
                                       mode=trt.ScaleMode.CHANNEL,
                                       shift=shift,
                                       scale=scale)
        bn = bn.get_output(0)

        if debug:
            print(f'BN_{layer_name}: ', bn.shape)
        return bn

    def add_convbn2d(self, input, weights, conv_name='', bn_name='', fuse=True, **kwargs) -> trt.ITensor:
        '''
        Add a fused convolution-batch normalization layer or separate convolution and batch normalization layers to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the layers will be added.
            input (trt.ITensor): The input tensor to be processed.
            weights (dict): A dictionary containing layer weights for convolution and batch normalization.
            conv_name (str, optional): The name of the convolution layer.
            bn_name (str, optional): The name of the batch normalization layer.
            fuse (bool, optional): If True, fuse convolution and batch normalization layers; otherwise, add them separately.
            **kwargs: Additional arguments for the `add_conv2d` function (stride, padding, etc.).

        Returns:
            trt.ITensor: The convolution or batch normalization layer(s) ITensor added to the network.

        Examples:
            To add a fused convolution-batch normalization layer to an input tensor:
            >>> input_tensor = input_tensor
            >>> layer_weights = {'conv.weight': conv_weight_tensor, 'conv.bias': conv_bias_tensor, 'bn.weight': bn_gamma_tensor, 'bn.bias': bn_beta_tensor, 'bn.running_mean': bn_mean_tensor, 'bn.running_var': bn_var_tensor}
            >>> fused_layer = add_convbn2d(network, input_tensor, layer_weights, conv_name='conv', bn_name='bn', fuse=True)

            To add separate convolution and batch normalization layers to an input tensor with custom stride and padding:
            >>> input_tensor = input_tensor
            >>> layer_weights = {'conv.weight': conv_weight_tensor, 'conv.bias': conv_bias_tensor, 'bn.weight': bn_gamma_tensor, 'bn.bias': bn_beta_tensor, 'bn.running_mean': bn_mean_tensor, 'bn.running_var': bn_var_tensor}
            >>> conv_layer = add_convbn2d(network, input_tensor, layer_weights, conv_name='conv', bn_name='bn', fuse=False, stride=(2, 2), padding=(1, 1), debug=True)    
        '''
        if fuse:
            # get conv weights
            if conv_name and not conv_name.endswith('.'):
                conv_name = f'{conv_name}.'
            w = weights[f'{conv_name}weight']
            bias_name = f'{conv_name}bias'
            b = weights[bias_name] if bias_name in weights else 0

            # get bn weights
            if bn_name and not bn_name.endswith('.'):
                bn_name = f'{bn_name}.'
            gamma = weights[f'{bn_name}weight']
            beta = weights[f'{bn_name}bias']
            mean = weights[f'{bn_name}running_mean']
            var = weights[f'{bn_name}running_var']
            eps = kwargs.pop('esp', 1e-5)
            std = np.sqrt(var + eps)

            # create new items to use existed `add_conv2d` function
            weights[f'fused.{conv_name}.weight'] = (
                w * broadcast_to(gamma / std, w.shape)
            )
            weights[f'fused.{conv_name}.bias'] = (
                b - mean) / std * gamma + beta

            layer = self.add_conv2d(input,
                                    weights,
                                    f'fused.{conv_name}',
                                    **kwargs)
        else:
            # fetch bn kwargs
            eps = kwargs.pop('esp', 1e-5)
            debug = kwargs.get('debug', False)

            layer = self.add_conv2d(input,
                                    weights,
                                    conv_name,
                                    debug=debug,
                                    **kwargs)
            layer = self.add_bn2d(layer,
                                  weights,
                                  bn_name,
                                  eps,
                                  debug=debug)
        return layer

    def add_activation(self,
                       input,
                       type=trt.ActivationType.RELU,
                       alpha=None,
                       beta=None,
                       debug=False) -> trt.ITensor:
        '''
        Add an activation layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the activation layer will be added.
            input (trt.ITensor): The input tensor to which the activation function will be applied.
            type (trt.ActivationType): The type of activation function (RELU, SIGMOID, TANH, CLIP, etc.).
            alpha (float, optional): The alpha value for the activation function (e.g., for Leaky ReLU).
            beta (float, optional): The beta value for the activation function (e.g., for SELU).
            debug (bool, optional): If True, print the shape of the output tensor.

        Returns:
            trt.ITensor: The activation ITensor added to the network.

        Examples:
            To add a ReLU activation layer to an input tensor:
            >>> relu_layer = add_activation(network, input_tensor, type=trt.ActivationType.RELU)

            To add a Leaky ReLU activation layer to an input tensor with a custom alpha value:
            >>> alpha_value = 0.01
            >>> leaky_relu_layer = add_activation(network, input_tensor, type=trt.ActivationType.LEAKY_RELU, alpha=alpha_value, debug=True)
        '''
        layer = self.network.add_activation(input=input, type=type)
        if alpha is not None:
            layer.alpha = alpha
        if beta is not None:
            layer.beta = beta

        layer = layer.get_output(0)
        return layer

    def add_pool2d(self,
                   input,
                   type=trt.PoolingType.AVERAGE,
                   kernel_size=2,
                   stride=2,
                   padding=0,
                   debug=False) -> trt.ITensor:
        '''
        Add a 2D pooling layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the pooling layer will be added.
            input (trt.ITensor): The input tensor to be pooled.
            type (trt.PoolingType): The type of pooling operation (MAX, AVERAGE, MAX_AVERAGE_BLEND).
            kernel_size (int or tuple): The size of the pooling window (e.g., 2 for a 2x2 window).
            stride (int or tuple): The stride for the pooling operation (e.g., 2 for a stride of 2).
            padding (int or tuple): The padding to be applied before pooling.
            debug (bool, optional): If True, print the shape of the output tensor.

        Returns:
            trt.ITensor: The pooling ITensor added to the network.

        Examples:
            To add a 2x2 max pooling layer to an input tensor:
            >>> max_pool_layer = add_pool2d(network, input_tensor, type=trt.PoolingType.MAX, kernel_size=2, stride=2)

            To add a 3x3 average pooling layer to an input tensor with custom kernel size and stride:
            >>> custom_kernel_size = (3, 3)
            >>> custom_stride = (2, 2)
            >>> avg_pool_layer = add_pool2d(network, input_tensor, type=trt.PoolingType.AVERAGE, kernel_size=custom_kernel_size, stride=custom_stride, debug=True)
        '''
        layer = self.network.add_pooling_nd(input=input,
                                            type=type,
                                            window_size=totuple(kernel_size))
        layer.stride_nd = totuple(stride)
        layer.padding_nd = totuple(padding)
        layer = layer.get_output(0)

        if debug:
            print(f'pool: {layer.shape}')
        return layer

    def add_reshape(self,
                    input,
                    shape,
                    debug=False) -> trt.ITensor:
        '''
        Add a reshape layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the reshape layer will be added.
            input (trt.ITensor): The input tensor to be reshaped.
            shape (tuple): The target shape for reshaping. (accept value 0, -1)
            debug (bool, optional): If True, print the shape of the output tensor.

        Returns:
            trt.ITensor: The reshape ITensor added to the network.

        Examples:
            To add a reshape layer with a target shape:
            >>> input_tensor = input_tensor
            >>> target_shape = (64, 3, 224, 224)
            >>> reshape_layer = add_reshape(network, input_tensor, target_shape, debug=True)

            To add a reshape layer with dynamic dimensions (e.g., -1):
            >>> input_tensor = input_tensor
            >>> dynamic_shape = (0, -1, 224, 224)
            >>> reshape_layer = add_reshape(network, input_tensor, dynamic_shape, debug=True)
        '''

        layer = self.network.add_shuffle(input)
        layer.reshape_dims = shape
        layer = layer.get_output(0)

        if debug:
            print(f'reshape: {layer.shape}')
        return layer

    def add_shuffle(self,
                    input,
                    p1=None,
                    shape=None,
                    p2=None,
                    debug=False) -> trt.ITensor:
        '''
        Add a shuffle layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the shuffle layer will be added.
            input (trt.ITensor): The input tensor to be shuffled.
            p1 (tuple, optional): The first permutation order for shuffling.
            shape (tuple, optional): The target shape for reshaping. Accepts values 0 and -1.
                Value 0: Copy the original dimension from the input.
                Value -1: Keep the remaining dimensions.
            p2 (tuple, optional): The second permutation order for shuffling.
            debug (bool, optional): If True, print the shape of the output tensor.

        Returns:
            trt.ITensor: The shuffle ITensor added to the network.

        Examples:
            To add a shuffle layer with a specified permutation and reshape:
            >>> input_tensor = input_tensor
            >>> permutation_1 = (0, 2, 1, 3)
            >>> target_shape = (0, -1, 224, 224)
            >>> permutation_2 = (0, 3, 2, 1)
            >>> shuffle_layer = add_shuffle(network, input_tensor, p1=permutation_1, shape=target_shape, p2=permutation_2, debug=True)    
        '''
        layer = self.network.add_shuffle(input)
        if p1 is not None:
            layer.first_transpose = p1

        if shape is not None:
            input_shape = input.shape
            new_shape = tuple(
                input_shape[i] if s == 0 else s for i, s in enumerate(shape)
            )
            layer.reshape_dims = new_shape

        if p2 is not None:
            layer.second_transpose = p2

        layer = layer.get_output(0)

        if debug:
            print(f'suffle layer: {layer.shape}')
        return layer

    def add_fully_connected(self,
                            input,
                            weights,
                            layer_name='',
                            debug=False) -> trt.ITensor:
        '''
        Add a fully connected layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the fully connected layer will be added.
            input (trt.ITensor): The input tensor for the fully connected layer.
            weights (dict): A dictionary containing the layer's weights and biases.
            layer_name (str, optional): The name of the fully connected layer.
            debug (bool, optional): If True, print the shape of intermediate tensors.

        Returns:
            trt.ITensor: The fully connected ITensor added to the network.

        Examples:
            To add a fully connected layer with input tensor 'input_tensor' and weights from 'weights_dict':
            >>> fc_layer = add_fully_connected(network, input_tensor, weights_dict, layer_name='fc1', debug=True)
        '''
        name = layer_name
        layer_name = f'{layer_name}.' if layer_name else ''
        w = weights[f'{layer_name}weight']
        bias_name = f'{layer_name}bias'
        b = weights[bias_name] if bias_name in weights else None

        num_output_channels, k = w.shape
        filter_const = self.network.add_constant(shape=(num_output_channels, k),
                                                 weights=w).get_output(0)
        layer = self.network.add_matrix_multiply(
            input0=input,
            op0=trt.MatrixOperation.NONE,
            input1=filter_const,
            op1=trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)

        if debug:
            print(f'{name} mul: {layer.shape}')

        if b is None:
            return layer

        bias_const = self.network.add_constant((1, num_output_channels),
                                               b).get_output(0)
        layer = self.network.add_elementwise(layer,
                                             bias_const,
                                             trt.ElementWiseOperation.SUM).get_output(0)
        if debug:
            print(f'{name} add: {layer.shape}')
        return layer

    def add_resize(self,
                   input,
                   shape=None,
                   scales=None,
                   mode=trt.ResizeMode.NEAREST,
                   debug=False) -> trt.ITensor:
        '''
        Add a resize layer to the TensorRT network.

        Note:
            Could only select either shape or scales
            shape and scales: without the requirement of specifying the dimensions of batch size and channels.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the resize layer will be added.
            input (trt.ITensor): The input tensor to be resized.
            shape (tuple, optional): The target shape for resizing. (row, col)
            scales (tuple, optional): The scaling factors for resizing. (row, col)
            mode (trt.ResizeMode, optional): The resize mode (e.g., NEAREST, LINEAR).
            debug (bool, optional): If True, print the shape of the output tensor.

        Returns:
            trt.ITensor: The resize ITensor added to the network.

        Examples:
            To add a resize layer with a target shape:
            >>> input_tensor = input_tensor
            >>> target_shape = (480, 640) # ROW, COL
            >>> resize_layer = add_resize(network, input_tensor, shape=target_shape, debug=True)

            To add a resize layer with scaling factors:
            >>> input_tensor = input_tensor
            >>> scaling_factors = (2.0, 2.0) # ROW, COL
            >>> resize_layer = add_resize(network, input_tensor, scales=scaling_factors, debug=True)
        '''
        layer = self.network.add_resize(input)
        layer.resize_mode = mode

        if shape is not None:
            # shape_tensor = shape_tensor * (1, 1, 0, 0) + (0, 0, r, c)
            shape_tensor = self.network.add_shape(input).get_output(0)
            s = shape_tensor.shape[0] - 2
            shape_tensor = self.add_elementwise(shape_tensor,
                                                (1, 1, *(0 for _ in range(s))),
                                                trt.ElementWiseOperation.PROD,
                                                dtype=np.int32)
            shape_tensor = self.add_elementwise(shape_tensor,
                                                (0, 0, *shape),
                                                trt.ElementWiseOperation.SUM,
                                                dtype=np.int32)
            layer.set_input(1, shape_tensor)

        elif scales is not None:
            scales = (1, 1, *scales)
            layer.scales = scales

        else:
            if debug:
                print('Should choose either shape or scales')
            return

        layer = layer.get_output(0)

        if debug:
            print('resize: ', layer.shape)
        return layer

    def add_cat(self, tensors, dim=0, debug=False) -> trt.ITensor:
        '''
        Add a concatenation layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the layer will be added.
            tensors (list): A list of input tensors to be concatenated. Limited up to 10000 tensors.
            dim (int): The dimension along which to concatenate the tensors.
            debug (bool): If True, print the shape of the output tensor.

        Returns:
            trt.ITensor: The concatenation ITensor added to the network.

        Examples:
            To concatenate a list of input tensors along dimension 1:
            >>> input_tensors = [tensor1, tensor2, tensor3]
            >>> cat_layer = add_cat(network, input_tensors, dim=1, debug=True)
        '''

        layer = self.network.add_concatenation(tensors)
        layer.axis = dim
        layer = layer.get_output(0)

        if debug:
            print(f'Concatenation: Output shape: {layer.shape}')
        return layer

    def add_reduce(self, input, op=trt.ReduceOperation.SUM, axes=(1, 0), keep_dims=True, debug=False) -> trt.ITensor:
        '''
        Add a reduction operation layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the layer will be added.
            input (trt.ITensor): The input tensor for the reduction operation.
            op (trt.ReduceOperation): The reduction operation type ('AVG', 'MAX', 'MIN', 'PROD', 'SUM').
            axes (tuple or int): The dimensions along which to perform the reduction, represented as a bitmask or integer.
            keep_dims (bool): If True, keep the reduced dimensions in the output tensor.
            debug (bool): If True, print the shape of the output tensor.

        Returns:
            trt.ITensor: The reduction ITensor added to the network.

        Examples:
            To add a SUM reduction layer along dimensions 1 and 0:
            >>> input_tensor = network.add_input("input", trt.DataType.FLOAT, (3, 4, 5))
            >>> reduce_layer = add_reduce(network, input_tensor, trt.ReduceOperation.SUM, axes=(1, 0), debug=True)
        '''

        axes = axes if isinstance(axes, int) else bitmask2int(axes)
        layer = self.network.add_reduce(
            input, op, axes, keep_dims).get_output(0)

        if debug:
            print(f'Reduce: Output shape: {layer.shape}')
        return layer

    def add_unary(self, input, op=trt.UnaryOperation.ABS, debug=True) -> trt.ITensor:
        '''
        Add a unary operation layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the layer will be added.
            input (trt.ITensor): The input tensor for the unary operation.
            op (trt.UnaryOperation): The unary operation type (
                                        'ABS', 'NEG', 'NOT', 'SIGN',
                                        'COS', 'SIN', 'TAN',
                                        'ACOS', 'ASIN', 'ATAN',
                                        'COSH', 'SINH',
                                        'ACOSH', 'ASINH', 'ATANH',
                                        'CEIL', 'FLOOR', 'ROUND',
                                        'LOG', 'EXP', 'SQRT', 
                                        'ERF', 'ISINF', 'RECIP',    
                                    ).
            debug (bool): If True, print the shape of the output tensor.

        Returns:
            trt.ITensor: The unary ITensor added to the network.

        Examples:
            To add an ABS (absolute) operation layer:
            >>> input_tensor = network.add_input("input", trt.DataType.FLOAT, (3, 4, 5))
            >>> unary_layer = add_unary(network, input_tensor, trt.UnaryOperation.ABS, debug=True)
        '''

        layer = self.network.add_unary(input, op).get_output(0)

        if debug:
            print(f'Unary Operation: Output shape: {layer.shape}')
        return layer

    def add_topk(self, input, op=trt.TopKOperation.MAX, k=1, axes=(1, 0), debug=False) -> tuple[trt.ITensor, trt.ITensor]:
        '''
        Add a TopK layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the layer will be added.
            input (trt.ITensor): The input tensor to the TopK layer.
            op (trt.TopKOperation): The operation type (MAX or MIN).
            k (int): The number of top elements to compute.
            axes (tuple): A tuple specifying the dimensions along which to perform the operation.
                        Represented as a bitmask, e.g., (1, 1, 0) for dimensions 1 and 2.
            debug (bool): If True, print the shape of the output tensor.

        Returns:
            index, value (tuple[trt.ITensor, trt.ITensor])

        Examples:
            To add a TopK layer that computes the maximum value and its index along dimension 1:
            >>> input_tensor = network.add_input("input", trt.DataType.FLOAT, (3, 4, 5))
            >>> topk_layer = add_topk(network, input_tensor, trt.TopKOperation.MAX, k=1, axes=(1,))
            >>> print('value:', topk_layer.get_output(0), 'index:', topk_layer.get_output(1))

            To add a TopK layer that computes the minimum value and its index along dimensions 1 and 2:
            >>> input_tensor = network.add_input("input", trt.DataType.FLOAT, (3, 4, 5))
            >>> topk_layer = add_topk(network, input_tensor, trt.TopKOperation.MIN, k=1, axes=(1, 2), debug=True)
            >>> print('value:', topk_layer.get_output(0), 'index:', topk_layer.get_output(1))
        '''

        axes = bitmask2int(axes)
        layer = self.network.add_topk(input, op, k, axes)

        if debug:
            print(f'TopK: Output shape: {layer.get_output(0).shape}')
        return layer.get_output(0), layer.get_output(1)

    def add_amax(self, input, dim=1, debug=False) -> tuple[trt.ITensor, trt.ITensor]:
        '''
        Add an Amax (maximum) operation layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the layer will be added.
            input (trt.ITensor): The input tensor to the Amax layer.
            dim (int): The dimension along which to compute the maximum value.
            debug (bool): If True, print the shape of the output tensor.

        Returns:
            index, value (tuple[trt.ITensor, trt.ITensor])

        Examples:
            To add an Amax layer that computes the maximum value along dimension 1:
            >>> input_tensor = network.add_input("input", trt.DataType.FLOAT, (3, 4, 5))
            >>> amax_layer = network.add_amax(input_tensor, dim=1, debug=True)
            >>> print('value:', amax_layer[0], 'index:', amax_layer[1])
        '''
        axes = dim2axes(dim)
        layer = self.add_topk(input,
                              trt.TopKOperation.MAX,
                              k=1,
                              axes=axes,
                              debug=debug)
        return layer

    def add_amin(self, input, dim=1, debug=False) -> tuple[trt.ITensor, trt.ITensor]:
        '''
        Add an Amin (minimum) operation layer to the TensorRT network.

        Args:
            network (trt.INetworkDefinition): The TensorRT network to which the layer will be added.
            input (trt.ITensor): The input tensor to the Amin layer.
            dim (int): The dimension along which to compute the minimum value.
            debug (bool): If True, print the shape of the output tensor.

        Returns:
            index, value (tuple[trt.ITensor, trt.ITensor])

        Examples:
            To add an Amin layer that computes the minimum value along dimension 0:
            >>> input_tensor = network.add_input("input", trt.DataType.FLOAT, (3, 4, 5))
            >>> amin_layer = network.add_amin(input_tensor, dim=0, debug=True)
            >>> print('value:', amin_layer[0], 'index:', amin_layer[1])
        '''
        axes = dim2axes(dim)
        layer = self.add_topk(input,
                              trt.TopKOperation.MIN,
                              k=1,
                              axes=axes,
                              debug=debug)
        return layer

    def add_einsum(self, inputs: list[trt.ITensor], eqation: str, debug=False) -> trt.ITensor:
        '''
        Adds an Einsum (Einstein summation convention) layer to the network. 

        Args:
            inputs (list[trt.ITensor]): The input tensors to the layer. It can only take a maximum of two inputs.
            equation (str): The Einsum equation of the layer.

        Returns:
            trt.ITensor: The ensum ITensor added to the network.

        Note: 
            This function is only available starting from TensorRT 8.2.0.
        '''
        if VERSION < '8.2':
            raise RuntimeError(
                f'Got TensorRT{VERSION}, '
                'Einstein summation is only available starting from TensorRT 8.2.0.'
            )
        if len(inputs) >= 3 and debug:
            print('Enum eqation can only take a maximum of two inputs.')

        layer = self.network.add_einsum(inputs, eqation).get_output(0)

        if debug:
            print(f'Emsum equation: {eqation} -> output shape: {layer.shape}')
        return layer
