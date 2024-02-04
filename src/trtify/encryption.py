
from __future__ import annotations

import base64
import os
import pkg_resources
import warnings

from .utils import Bytes

__all__ = [
    'Cryptography',
]


class ValidToken(Bytes):
    def __new__(cls, token=b'', file_name='', **kwargs):
        if token and ValidToken.is_valid(token):
            return super().__new__(cls, token, **kwargs)
        elif token:
            warnings.warn(f'{token} is not a valid token, return generated token instead')
        return cls.read_token(file_name) if file_name else cls.generate_token()

    @staticmethod
    def is_valid(token) -> bool:
        try: token = base64.urlsafe_b64decode(token)
        except: return False
        if len(token) != 32: return False
        return True

    @classmethod
    def read_token(cls, filename='file.bin'):
        token = cls.read_bin(filename=filename)
        return cls(token)
    
    @classmethod
    def generate_token(cls):
        return cls(base64.urlsafe_b64encode(os.urandom(32)))
        

class Cryptography:
    def __new__(cls, *args, **kwargs):
        if 'cryptography' not in (pkg.key for pkg in pkg_resources.working_set):
            from .utils import Null
            warnings.warn(
                'Dependencies are not installed, '
                'using dummy class instead',
                RuntimeWarning)
            return Null()

        from cryptography.fernet import Fernet as _Fernet

        class Fernet(_Fernet):
            def encrypt(self, data: bytes) -> bytes: return Bytes(super().encrypt(data))
            def decrypt(self, token: bytes | str, ttl: int | None = None) -> bytes:
                return Bytes(super().decrypt(token, ttl))

        cls.Fernet = Fernet
        return object.__new__(cls)

    def __init__(self, token=b'', keyfname=''):
        self._token = ValidToken(token, keyfname)
        self.fernet = self.Fernet(self.token)

    @property
    def token(self):
        return self._token
    
    @token.setter
    def token(self, token: bytes):
        self._token = ValidToken(token)

    def encrypt(self, binary: bytes) -> bytes:
        if not isinstance(binary, bytes):
            try:
                binary = Bytes(binary)
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
        self._token = ValidToken.read_token(keyfname)
        return self.token

    def export_key(self, keyfname='key.bin'):
        self.token.to_bin(keyfname)


def read_key(keyfname='key.bin'):
    with open(keyfname, 'rb') as f:
        token = f.read()
    return token


def export_key(token, keyfname='key.bin'):
    with open(keyfname, 'wb') as key_file:
        key_file.write(token)
