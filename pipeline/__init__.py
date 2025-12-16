"""
Pipeline package for advanced RAG-based privacy policy analysis.
"""
from .chunker import TokenAwareChunker, chunk_privacy_policy
from .embedder import CachedEmbedder, embed_privacy_chunks
from .indexer import ChromaIndexer, BM25Retriever, HybridRetriever
from .reporter import MarkdownReporter, generate_privacy_report
from .rag_pipeline import PrivacyPolicyPipeline

__all__ = [
    'TokenAwareChunker',
    'chunk_privacy_policy',
    'CachedEmbedder',
    'embed_privacy_chunks',
    'ChromaIndexer',
    'BM25Retriever',
    'HybridRetriever',
    'MarkdownReporter',
    'generate_privacy_report',
    'PrivacyPolicyPipeline'
]
