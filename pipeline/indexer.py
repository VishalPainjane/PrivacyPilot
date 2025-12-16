"""
Vector store indexer with support for Chroma (dev) and Elasticsearch (BM25).
Hybrid retrieval combining dense vectors + BM25 lexical search.
"""
import os
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np


class ChromaIndexer:
    """
    Chroma vector store indexer for development/prototyping.
    Lightweight, local, no external dependencies.
    """
    
    def __init__(
        self,
        collection_name: str = "privacy_policies",
        persist_directory: str = ".chroma_db",
        embedding_dim: int = 384
    ):
        """
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist Chroma DB
            embedding_dim: Dimension of embeddings
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embedding_dim = embedding_dim
        
        # Initialize client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        
        print(f"Chroma initialized: {self.collection.count()} documents")
    
    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Add chunks to Chroma collection.
        
        Args:
            chunks: List of chunks with 'embedding', 'text', 'chunk_id'
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Prepare data
        ids = [str(chunk.get('chunk_id', i)) for i, chunk in enumerate(chunks)]
        embeddings = [chunk['embedding'].tolist() for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        
        # Prepare metadata
        metadatas = []
        for chunk in chunks:
            meta = {
                "url": chunk.get('metadata', {}).get('url', '') or '',
                "header": chunk.get('metadata', {}).get('header', '') or 'N/A',
                "tokens": int(chunk.get('tokens', 0)),
                "chunk_id": int(chunk.get('chunk_id', 0))
            }
            metadatas.append(meta)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of results with scores
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_dict
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'chunk_id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'metadata': results['metadatas'][0][i]
            })
        
        return formatted
    
    def delete_all(self):
        """Delete all documents from collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection cleared")
    
    def count(self) -> int:
        """Get total document count"""
        return self.collection.count()


class BM25Retriever:
    """
    BM25 lexical retriever for keyword-based search.
    Fallback to simple TF-IDF if Elasticsearch not available.
    """
    
    def __init__(self, use_elasticsearch: bool = False):
        """
        Args:
            use_elasticsearch: Use Elasticsearch backend (requires ES server)
        """
        self.use_elasticsearch = use_elasticsearch
        
        if use_elasticsearch:
            try:
                from elasticsearch import Elasticsearch
                self.es_client = Elasticsearch(['http://localhost:9200'])
                self.index_name = "privacy_policies"
                print("Elasticsearch BM25 enabled")
            except ImportError:
                print("Elasticsearch not available, falling back to simple BM25")
                self.use_elasticsearch = False
        
        if not use_elasticsearch:
            # Simple in-memory BM25
            self.documents = []
            self.doc_metadata = []
    
    def add_chunks(self, chunks: List[Dict]):
        """Add chunks for BM25 indexing"""
        if self.use_elasticsearch:
            self._add_to_elasticsearch(chunks)
        else:
            self.documents.extend([chunk['text'] for chunk in chunks])
            self.doc_metadata.extend(chunks)
    
    def _add_to_elasticsearch(self, chunks: List[Dict]):
        """Add chunks to Elasticsearch"""
        from elasticsearch.helpers import bulk
        
        actions = []
        for chunk in chunks:
            action = {
                "_index": self.index_name,
                "_id": str(chunk.get('chunk_id', '')),
                "_source": {
                    "text": chunk['text'],
                    "metadata": chunk.get('metadata', {})
                }
            }
            actions.append(action)
        
        bulk(self.es_client, actions)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        BM25 search for query.
        
        Args:
            query: Search query text
            top_k: Number of results
            
        Returns:
            List of results with BM25 scores
        """
        if self.use_elasticsearch:
            return self._search_elasticsearch(query, top_k)
        else:
            return self._search_simple(query, top_k)
    
    def _search_simple(self, query: str, top_k: int) -> List[Dict]:
        """Simple TF-IDF based search as fallback"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        if not self.documents:
            return []
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english')
        doc_vectors = vectorizer.fit_transform(self.documents)
        query_vector = vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, doc_vectors)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only return relevant results
                results.append({
                    'chunk_id': self.doc_metadata[idx].get('chunk_id', idx),
                    'text': self.documents[idx],
                    'score': float(similarities[idx]),
                    'metadata': self.doc_metadata[idx].get('metadata', {})
                })
        
        return results
    
    def _search_elasticsearch(self, query: str, top_k: int) -> List[Dict]:
        """Search using Elasticsearch BM25"""
        response = self.es_client.search(
            index=self.index_name,
            body={
                "query": {
                    "match": {
                        "text": query
                    }
                },
                "size": top_k
            }
        )
        
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'chunk_id': hit['_id'],
                'text': hit['_source']['text'],
                'score': hit['_score'],
                'metadata': hit['_source'].get('metadata', {})
            })
        
        return results


class HybridRetriever:
    """
    Hybrid retriever combining dense vectors (Chroma) and BM25.
    Weighted fusion: score = α * bm25_score + β * vector_score
    Recommended: α=0.4, β=0.6
    """
    
    def __init__(
        self,
        vector_indexer: ChromaIndexer,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.4,
        beta: float = 0.6
    ):
        """
        Args:
            vector_indexer: Vector store indexer
            bm25_retriever: BM25 retriever
            alpha: Weight for BM25 scores
            beta: Weight for vector scores
        """
        self.vector_indexer = vector_indexer
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        self.beta = beta
    
    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 and vector similarity.
        
        Args:
            query: Query text for BM25
            query_embedding: Query embedding for vector search
            top_k: Total results to return
            
        Returns:
            Merged and re-ranked results
        """
        # Get results from both retrievers
        vector_results = self.vector_indexer.search(query_embedding, top_k=top_k * 2)
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
        
        # Normalize scores to [0, 1]
        vector_results = self._normalize_scores(vector_results)
        bm25_results = self._normalize_scores(bm25_results)
        
        # Merge results
        merged = {}
        
        for result in vector_results:
            chunk_id = result['chunk_id']
            merged[chunk_id] = {
                **result,
                'vector_score': result['score'],
                'bm25_score': 0.0
            }
        
        for result in bm25_results:
            chunk_id = result['chunk_id']
            if chunk_id in merged:
                merged[chunk_id]['bm25_score'] = result['score']
            else:
                merged[chunk_id] = {
                    **result,
                    'vector_score': 0.0,
                    'bm25_score': result['score']
                }
        
        # Calculate hybrid scores
        for chunk_id in merged:
            vec_score = merged[chunk_id]['vector_score']
            bm_score = merged[chunk_id]['bm25_score']
            merged[chunk_id]['hybrid_score'] = self.alpha * bm_score + self.beta * vec_score
        
        # Sort by hybrid score
        ranked = sorted(merged.values(), key=lambda x: x['hybrid_score'], reverse=True)
        
        return ranked[:top_k]
    
    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """Normalize scores to [0, 1] range"""
        if not results:
            return results
        
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for r in results:
                r['score'] = 1.0
        else:
            for r in results:
                r['score'] = (r['score'] - min_score) / (max_score - min_score)
        
        return results
