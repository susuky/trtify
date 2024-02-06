
import tensorrt as trt
TRT_VERSION = trt.__version__

from .builder.builder import *
from .network_definition import *
from .runtime.runtime import *