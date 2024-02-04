
import datetime
import contextlib
import json
import numpy as np
import operator
import pickle
import time

__all__ = [
    'check_dim',
    'Null',
    'totuple',
    'dim2axes',
    'bitmask2int',
    'broadcast_to',
]

DATETIME_FORMAT = '%Y-%m-%d_%H-%M-%S'


def save_pickle(data, filename=None):
    """
    Save the given data as a pickle file.

    Parameters:
        data (Any): The data to be saved.
        filename (Optional[str]): The name of the file to be saved. If not provided, a timestamp-based filename will be generated.

    Returns:
        None
    """
    filename = filename or datetime.datetime.now().strftime(DATETIME_FORMAT)
    if not str(filename).endswith('.pkl'): 
        filename = f'{filename}.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    """
    Load and return the object stored in a pickle file.

    Args:
        filename (str): The path to the pickle file to be loaded.

    Returns:
        The object stored in the pickle file.

    Raises:
        FileNotFoundError: If the specified pickle file does not exist.
        pickle.UnpicklingError: If there is an error while unpickling the object.

    """
    if not str(filename).endswith('.pkl'): 
        filename = f'{filename}.pkl'
        
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_json(data, filename=None, indent=4, encoding='utf-8'):
    filename = filename or datetime.datetime.now().strftime(DATETIME_FORMAT)
    if not str(filename).endswith('.json'): 
        filename = f'{filename}.json'
        
    with open(filename, 'w', encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filename):
    if not str(filename).endswith('.json'): 
        filename = f'{filename}.json'
        
    with open(filename, 'r') as f:
        return json.load(f)


def check_dim(a, b, op='<='):
    '''
    op:
        1. `operator.lt(a, b)`: `<` (less than) operator.
        2. `operator.le(a, b)`:  `<=` (less than or equal to) operator.
        3. `operator.eq(a, b)`:  `==` (equal to) operator.
        4. `operator.ne(a, b)`:  `!=` (not equal to) operator.
        5. `operator.gt(a, b)`:  `>` (greater than) operator.
        6. `operator.ge(a, b)`:  `>=` (greater than or equal to) operator.
    '''
    if len(a) != len(b):
        return False
    mapper = {
        '<': operator.lt,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne,
        '>': operator.lt,
        '>=': operator.le,
    }

    operation = mapper.get(op, getattr(operator, op, None))
    if operation is None:
        raise ValueError(f'{op} is not supported')

    return all(operation(i, j) for i, j in zip(a, b))


class Null:
    '''
    This class essentially acts as a universal "do nothing" placeholder or null object. 
    '''
    def __init__(self, *args, **kwargs): ...
    def __getattribute__(self, x): return self
    def __getitem__(self, idx): return self
    def __iter__(self): return iter(range(0))
    def __next__(self): return self
    def __len__(self): return 0
    def __call__(self, *args, **kwargs): return self
    def __repr__(self): return ''


def totuple(v):
    """
    Convert a single value or a single-element tuple into a tuple with two elements.

    Args:
        v: The input value to be converted.

    Returns:
        tuple: A tuple with two elements. If the input is already a tuple, it is returned unchanged.
    """
    if not isinstance(v, tuple):
        v = (v, ) * 2
    return v


def dim2axes(dim: int) -> tuple:
    '''
    Convert an integer into a tuple of binary digits representing a bitmask with a single 1 at position `dim`.

    Args:
        dim (int): The position at which to set the bit to 1.

    Returns:
        tuple: A tuple containing binary digits representing a bitmask with a single 1 at the specified position.

    Examples:
        >>> dim2axes(2)
        (1, 0, 0)  # Bitmask with a single 1 at position 2
        >>> dim2axes(1)
        (1, 0)
    '''
    return tuple([1, *(0 for _ in range(dim))])


def bitmask2int(bitmask: tuple) -> int:
    '''
    Convert a binary tuple to an integer.

    Args:
        bitmask (tuple[int]): A tuple of binary digits (0 or 1).

    Returns:
        int: The integer representation of the binary tuple.

    Examples:
        >>> bitmask2int((1, 1, 1))
        7  # (2^2 + 2^1 + 1)
        >>> bitmask2int((1, 0, 1))
        5  # (2^2 + 1)
    '''
    return int(''.join(map(str, bitmask)), base=2)


def broadcast_to(array, shape, broadcast=False):
    '''
    Pad and optionally broadcast an input array to match a target shape.

    Args:
        array (array-like): The input array to be padded and/or broadcasted.
        shape (tuple): The target shape that the input array should match.
        broadcast (bool, optional): If True, the input array will be broadcasted to match the target shape.
                                    If False, only dummy dimensions will be padded without broadcasting.

    Returns:
        numpy.ndarray: The padded and/or broadcasted array.

    Examples:
        To pad an array of shape (1, 3) to shape (1, 3, 4, 4):
        >>> array = [[0.485, 0.456, 0.406]]
        >>> target_shape = (1, 3, 4, 4)
        >>> result = broadcast_to(array, target_shape)
        >>> print(result.shape)  # Output: (1, 3, 4, 4)

        To pad an array of shape (3,) to shape (1, 3, 1, 1) without broadcasting:
        >>> array = [0.485, 0.456, 0.406]
        >>> target_shape = (1, 3, 224, 224)
        >>> result = broadcast_to(array, target_shape, broadcast=False)
        >>> print(result.shape)  # Output: (1, 3, 1, 1)

        To broadcast an array of shape (1, 3) to shape (2, 3, 4, 4):
        >>> array = [[0.485, 0.456, 0.406]]
        >>> target_shape = (2, 3, 4, 4)
        >>> result = broadcast_to(array, target_shape, broadcast=True)
        >>> print(result.shape)  # Output: (2, 3, 4, 4)
    '''

    try:
        if isinstance(array, (int, float)):
            for i in range(len(shape)):
                array = [array]
            array_pad = np.array(array)
        else:
            array_pad = np.array(array)
            if len(array_pad.shape) < len(shape):
                index = np.argmax(array_pad.shape)
                size = array_pad.shape[index]

                if size != 1:
                    for idx, s in enumerate(shape):
                        if s == size:
                            break

                    prepad = range(idx - index)
                    postpad = range(idx + 1, len(shape))

                    array_pad = np.expand_dims(array_pad,
                                               axis=tuple([*prepad, *postpad]))

        if broadcast:
            array_pad = np.broadcast_to(array_pad, shape)

    except ValueError:
        return array

    return array_pad


class Timer(contextlib.ContextDecorator):
    '''
    Time Profile class. Usage: @Profile() decorator or 'with Profile()
    Note: it does not contain synchronizing cuda stream 
    '''
    def __init__(self, t=0.0):
        '''
        Initialize Timer object.
        
        Parameters:
            t (float): The initial time value. Default is 0.0.
        '''
        self.t = t

    def __enter__(self):
        '''
        Enter the timer context.
        
        Returns:
            Timer: The Timer object itself.
        '''
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        '''
        Exit the timer context.
        
        Parameters:
            type: The exception type, if an exception occurred.
            value: The exception value, if an exception occurred.
            traceback: The traceback object, if an exception occurred.
        '''
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        '''
        Get the current time.
        
        Returns:
            float: The current time.
        '''
        return time.time()


class Bytes(bytes):
    @classmethod
    def read_bin(cls, filename='file.bin'):
        with open(filename, 'rb') as f:
            return cls(f.read())
        
    def to_bin(self, filename='file.bin'):
        with open(filename, 'wb') as f:
            f.write(self)
