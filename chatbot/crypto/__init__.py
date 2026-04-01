"""
Q-LREF Cryptographic Module for Ayurvedic Chatbot.

Provides post-quantum lattice-based key exchange and
AES-256-GCM encryption for secure communication.
"""

from .session import QLREFSession

__all__ = ["QLREFSession"]
