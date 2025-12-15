"""
科学词表扩展模块

提供对 SMILES、FASTA、IUPAC 等科学符号的词表扩展支持
"""

from .vocab_extension import (
    initialize_new_embeddings,
    extend_model_embeddings,
    load_science_tokenizer
)

__all__ = [
    'initialize_new_embeddings',
    'extend_model_embeddings',
    'load_science_tokenizer',
]
