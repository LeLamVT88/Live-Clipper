import base64
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from src.config import WORKER_DEFAULT_CONFIG as config

class AESCipher:
    def __init__(self):
        _raw_key = config.get('encrypt_key', 'tv360')
        if not _raw_key:
            raise EnvironmentError("ENCRYPTION_KEY không được để trống")
        try:
            self._key = base64.b64decode(_raw_key)
        except Exception:
            # Fallback if key is not base64: pad or truncate key to 32 bytes for AES-256
            self._key = _raw_key.encode('utf-8').ljust(32, b'\0')[:32]

    def decrypt(self, encoded_ciphertext: str) -> str:
        if not encoded_ciphertext:
            return ""
        try:
            raw = base64.b64decode(encoded_ciphertext)
            if len(raw) < 16:
                return encoded_ciphertext
            iv = raw[:16]
            ciphertext = raw[16:]
            cipher = Cipher(algorithms.AES(self._key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            unpadder = padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
            return plaintext.decode('utf-8')
        except Exception:
            # If decryption fails (e.g. text is plaintext), return original string
            return encoded_ciphertext

    def encrypt(self, plaintext: str) -> str:
        if not plaintext:
            return ""
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self._key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext.encode('utf-8')) + padder.finalize()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        return base64.b64encode(iv + ciphertext).decode('utf-8')
