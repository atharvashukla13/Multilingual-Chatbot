"""
AES-256-GCM Encryption/Decryption wrapper.

Used for symmetric encryption after Q-LREF key exchange
establishes a shared secret.
"""

import os
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

NONCE_SIZE = 12  # 96-bit nonce for GCM


def encrypt(plaintext_bytes, key):
    """
    AES-256-GCM encrypt.

    Returns:
        bytes: nonce (12 bytes) + ciphertext + tag (16 bytes)
    """
    nonce = os.urandom(NONCE_SIZE)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, None)
    return nonce + ciphertext


def decrypt(encrypted_blob, key):
    """
    AES-256-GCM decrypt.

    Returns:
        bytes: decrypted plaintext
    """
    nonce = encrypted_blob[:NONCE_SIZE]
    ciphertext = encrypted_blob[NONCE_SIZE:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)


def encrypt_string(text, key):
    """Encrypt a UTF-8 string, return bytes."""
    return encrypt(text.encode("utf-8"), key)


def decrypt_string(encrypted_blob, key):
    """Decrypt bytes back to a UTF-8 string."""
    return decrypt(encrypted_blob, key).decode("utf-8")


def bytes_to_hex_preview(data, max_len=64):
    """Convert bytes to hex string preview (for UI display)."""
    hex_str = data.hex()
    if len(hex_str) > max_len:
        return hex_str[:max_len] + "..."
    return hex_str


def bytes_to_base64_preview(data, max_len=80):
    """Convert bytes to base64 string preview (for UI display)."""
    b64 = base64.b64encode(data).decode("ascii")
    if len(b64) > max_len:
        return b64[:max_len] + "..."
    return b64
