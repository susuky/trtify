
import tensorrt as trt
TRT_VERSION = trt.__version__

from .builder import *
from .network_definition import *
from .runtime import *