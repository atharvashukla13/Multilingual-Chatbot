"""
Q-LREF: Quantum-Lattice Resilient Exchange Framework

Core LWE (Learning With Errors) based key exchange protocol.
Implements lattice-based post-quantum key generation, shared secret
derivation, and reconciliation.

Reference: Gohin Balakrishnan et al., IEEE ICCSP 2025
"""

import numpy as np
import hashlib


DEFAULT_PARAMS = {
    "n": 512,       # Lattice dimension
    "q": 12289,     # Prime modulus (same as NewHope)
    "sigma": 3.2,   # Gaussian noise std dev
    "lambda": 128,  # Security parameter (bits)
}


def _sample_gaussian(size, sigma, q):
    """Sample from discrete Gaussian distribution over Z_q."""
    samples = np.round(np.random.normal(0, sigma, size)).astype(np.int64)
    return samples % q


def _sample_uniform(shape, q):
    """Sample uniformly from Z_q."""
    return np.random.randint(0, q, size=shape, dtype=np.int64)


def generate_matrix(n, q):
    """Generate shared public matrix A in Z_q^{n x n}."""
    return _sample_uniform((n, n), q)


def generate_keypair(A, n, q, sigma):
    """
    LWE Key Generation.

    KeyGen(lambda, n, q, chi):
        s <- chi^n          (private key from Gaussian)
        e <- chi^n          (error vector from Gaussian)
        p = A . s + e mod q (public key)

    Returns:
        public_key: p = A*s + e (mod q), shape (n,)
        private_key: s, shape (n,)
    """
    s = _sample_gaussian(n, sigma, q)
    e = _sample_gaussian(n, sigma, q)
    p = (A @ s + e) % q
    return p, s


def derive_shared_secret(my_private_key, their_public_key, q, sigma):
    """
    Derive approximate shared secret using LWE.

    K = s_mine * p_theirs + e' (mod q)

    Both parties compute this — results are approximately equal
    (differ only by small noise).
    """
    n = len(my_private_key)
    e_prime = _sample_gaussian(n, sigma, q)
    raw_secret = (my_private_key * their_public_key + e_prime) % q
    return raw_secret


def reconcile(raw_secret, q):
    """
    Reconciliation: extract shared bits from approximate secret.

    Maps each coefficient to 0 or 1 based on proximity to 0 or q/2.
    Eliminates the small noise difference between two parties.
    """
    threshold = q // 4
    reconciled = np.where(
        (raw_secret >= threshold) & (raw_secret < 3 * threshold),
        1, 0
    ).astype(np.uint8)
    return reconciled


def reconciled_to_aes_key(reconciled_bits):
    """Convert reconciled bit vector to 256-bit AES key via SHA-256."""
    return hashlib.sha256(reconciled_bits.tobytes()).digest()


def hash_secret(data):
    """SHA-256 hash for verification exchange."""
    if isinstance(data, np.ndarray):
        data = data.tobytes()
    return hashlib.sha256(data).hexdigest()


def get_public_key_size_bytes(n):
    """Public key size in bytes (int16 per coefficient)."""
    return n * 2


def serialize_public_key(public_key):
    """Serialize public key to bytes."""
    return public_key.astype(np.int16).tobytes()


def deserialize_public_key(key_bytes):
    """Deserialize public key from bytes."""
    return np.frombuffer(key_bytes, dtype=np.int16).astype(np.int64)
