
import pkg_resources
import warnings

__all__ = [
    'Cryptography',
]


class Cryptography:
    def __new__(cls, *args, **kwargs):
        if 'cryptography' not in (pkg.key for pkg in pkg_resources.working_set):
            from .utils import Null
            warnings.warn(
                'Dependencies are not installed, '
                'using dummy class instead',
                RuntimeWarning)
            return Null()

        from cryptography.fernet import Fernet
        cls.Fernet = Fernet
        return object.__new__(cls)

    def __init__(self, token=None, keyfname=''):
        '''
        token (bytes) = None
        key_fname (str) = None
        '''
        if token:
            self.token = token
        elif keyfname:
            self.token = self.read_key(keyfname)
        else:
            self.token = self.Fernet.generate_key()

        self.fernet = self.Fernet(self.token)

    def encrypt(self, binary: bytes) -> bytes:
        if not isinstance(binary, bytes):
            try:
                binary = bytes(binary)
            except:
                warnings.warn(
                    f'Cannot convert {binary} into bytes',
                    RuntimeWarning
                )
                binary = b''
        return self.fernet.encrypt(binary)

    def decrypt(self, binary: bytes) -> bytes:
        if not isinstance(binary, bytes):
            try:
                binary = bytes(binary)
            except:
                warnings.warn(
                    f'Cannot convert {binary} into bytes',
                    RuntimeWarning
                )
                binary = b''
        return self.fernet.decrypt(binary)

    def read_key(self, keyfname='key.bin'):
        return read_key(keyfname=keyfname)

    def export_key(self, keyfname='key.bin'):
        export_key(self.token, keyfname)


def read_key(keyfname='key.bin'):
    with open(keyfname, 'rb') as f:
        token = f.read()
    return token


def export_key(token, keyfname='key.bin'):
    with open(keyfname, 'wb') as key_file:
        key_file.write(token)
