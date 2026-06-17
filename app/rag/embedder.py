"""
Backward-compatible re-exports from the unified embedder module.
"""

from app.embedder import embed_text, encode_texts, get_model

__all__ = ["embed_text", "encode_texts", "get_model"]
