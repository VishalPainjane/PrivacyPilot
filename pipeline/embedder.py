"""
Batch embedder with SHA256 caching for efficient vector generation.
Uses sentence-transformers with batching and persistence.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class CachedEmbedder:
    """
    Embedding generator with local caching and batch processing.
    Recommended model: all-MiniLM-L6-v2 (384 dimensions, fast)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = ".embedding_cache",
        batch_size: int = 32,
        normalize: bool = True
    ):
        """
        Args:
            model_name: HuggingFace model name
            cache_dir: Directory for embedding cache
            batch_size: Batch size for encoding
            normalize: Normalize embeddings to unit vectors
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.normalize = normalize
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
        # Load cache index
        self.cache_index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, str]:
        """Load cache index mapping hashes to file paths"""
        if self.cache_index_path.exists():
            with open(self.cache_index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        with open(self.cache_index_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache_index, f)
    
    def _hash_text(self, text: str) -> str:
        """Generate SHA256 hash for text"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _get_cached_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding if exists"""
        if text_hash in self.cache_index:
            cache_file = self.cache_dir / self.cache_index[text_hash]
            if cache_file.exists():
                return np.load(cache_file)
        return None
    
    def _cache_embedding(self, text_hash: str, embedding: np.ndarray):
        """Save embedding to cache"""
        cache_file = self.cache_dir / f"{text_hash}.npy"
        np.save(cache_file, embedding)
        self.cache_index[text_hash] = f"{text_hash}.npy"
        self._save_cache_index()
    
    def embed_single(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Input text
            use_cache: Whether to use cache
            
        Returns:
            Embedding vector as numpy array
        """
        text_hash = self._hash_text(text)
        
        # Check cache
        if use_cache:
            cached = self._get_cached_embedding(text_hash)
            if cached is not None:
                return cached
        
        # Generate embedding
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        
        # Cache result
        if use_cache:
            self._cache_embedding(text_hash, embedding)
        
        return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for batch of texts with caching.
        
        Args:
            texts: List of input texts
            use_cache: Whether to use cache
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            text_hash = self._hash_text(text)
            if use_cache:
                cached = self._get_cached_embedding(text_hash)
                if cached is not None:
                    embeddings.append((i, cached))
                    continue
            
            uncached_texts.append(text)
            uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            print(f"Generating embeddings for {len(uncached_texts)} texts ({len(texts) - len(uncached_texts)} cached)")
            
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=show_progress
            )
            
            # Cache new embeddings
            if use_cache:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    text_hash = self._hash_text(text)
                    self._cache_embedding(text_hash, embedding)
            
            # Add to results
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, embedding))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    
    def embed_chunks(
        self,
        chunks: List[Dict],
        text_key: str = "text",
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Embed chunks and add embedding to each chunk dict.
        
        Args:
            chunks: List of chunk dictionaries
            text_key: Key containing text to embed
            use_cache: Whether to use cache
            
        Returns:
            Chunks with added 'embedding' field
        """
        texts = [chunk[text_key] for chunk in chunks]
        embeddings = self.embed_batch(texts, use_cache=use_cache)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
            chunk['embedding_dim'] = self.embedding_dim
            chunk['embedding_model'] = self.model_name
        
        return chunks
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.cache_index = {}
            print("Cache cleared")
    
    def cache_stats(self) -> Dict:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.npy"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cached_embeddings": len(self.cache_index),
            "cache_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }


# Convenience function
def embed_privacy_chunks(
    chunks: List[Dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: str = ".embedding_cache"
) -> List[Dict]:
    """
    Quick function to embed chunks with caching.
    
    Args:
        chunks: List of chunk dictionaries with 'text' field
        model_name: Model to use
        cache_dir: Cache directory
        
    Returns:
        Chunks with embeddings added
    """
    embedder = CachedEmbedder(model_name=model_name, cache_dir=cache_dir)
    return embedder.embed_chunks(chunks)
