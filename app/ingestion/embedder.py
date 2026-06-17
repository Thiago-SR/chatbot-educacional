"""
Backward-compatible re-exports from the unified embedder module.
"""

from app.embedder import DEFAULT_BATCH_SIZE, encode_texts, get_model

__all__ = ["DEFAULT_BATCH_SIZE", "encode_texts", "get_model"]
