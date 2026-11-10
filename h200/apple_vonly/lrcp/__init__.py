"""Reusable LRCP core exports.

Keep package import lightweight: task-specific adapters such as HF/CLIP/SegFormer
may require optional dependencies, so import them from their submodules directly.
"""

from .headwise import LRCP
from .sequence import SequenceSelfAttention, SequenceTransformerBlock

__all__ = [
    "LRCP",
    "SequenceSelfAttention",
    "SequenceTransformerBlock",
]
