"""
Main RAG pipeline orchestrator.
Integrates chunking, embedding, retrieval, and analysis.
"""
import json
import time
from typing import List, Dict, Optional
from pathlib import Path

from .chunker import TokenAwareChunker
from .embedder import CachedEmbedder
from .indexer import ChromaIndexer, BM25Retriever, HybridRetriever
from .reporter import MarkdownReporter


class PrivacyPolicyPipeline:
    """
    End-to-end RAG pipeline for privacy policy analysis.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 100,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_hybrid: bool = True,
        alpha: float = 0.4,  # BM25 weight
        beta: float = 0.6,   # Vector weight
        cache_dir: str = ".embedding_cache",
        chroma_dir: str = ".chroma_db",
        template_path: str = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            chunk_size: Tokens per chunk
            overlap: Overlap tokens
            embedding_model: Model for embeddings
            use_hybrid: Use hybrid retrieval (BM25 + vectors)
            alpha: BM25 weight in hybrid search
            beta: Vector weight in hybrid search
            cache_dir: Embedding cache directory
            chroma_dir: Chroma database directory
            template_path: Path to prompt template JSON
        """
        print("Initializing Privacy Policy RAG Pipeline...")
        
        # Initialize components
        self.chunker = TokenAwareChunker(
            chunk_tokens=chunk_size,
            overlap_tokens=overlap,
            preserve_headers=True
        )
        
        self.embedder = CachedEmbedder(
            model_name=embedding_model,
            cache_dir=cache_dir,
            batch_size=32
        )
        
        self.vector_store = ChromaIndexer(
            collection_name="privacy_policies",
            persist_directory=chroma_dir,
            embedding_dim=self.embedder.embedding_dim
        )
        
        self.bm25 = BM25Retriever(use_elasticsearch=False)
        
        # Setup retriever
        self.use_hybrid = use_hybrid
        if use_hybrid:
            self.retriever = HybridRetriever(
                vector_indexer=self.vector_store,
                bm25_retriever=self.bm25,
                alpha=alpha,
                beta=beta
            )
        else:
            self.retriever = self.vector_store
        
        # Load template
        self.template_path = template_path or str(Path(__file__).parent / "prompt_template.json")
        self.template = self._load_template()
        
        self.reporter = MarkdownReporter(template_path=self.template_path)
        
        # Metadata
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_model = embedding_model
        
        print("Pipeline initialized successfully!")
    
    def _load_template(self) -> Dict:
        """Load prompt template"""
        if Path(self.template_path).exists():
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def index_document(self, text: str, url: str) -> Dict:
        """
        Index a privacy policy document.
        
        Args:
            text: Full policy text
            url: Source URL
            
        Returns:
            Indexing statistics
        """
        print(f"\nIndexing document from: {url}")
        start_time = time.time()
        
        # Step 1: Chunk document
        print("Step 1/3: Chunking document...")
        chunks = self.chunker.chunk_document(text, url)
        print(f"  Created {len(chunks)} chunks")
        
        # Step 2: Generate embeddings
        print("Step 2/3: Generating embeddings...")
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)
        
        # Step 3: Index
        print("Step 3/3: Indexing chunks...")
        
        # Add to vector store
        vector_count = self.vector_store.add_chunks(chunks_with_embeddings)
        
        # Add to BM25
        self.bm25.add_chunks(chunks_with_embeddings)
        
        elapsed = time.time() - start_time
        
        stats = {
            'url': url,
            'total_chunks': len(chunks),
            'indexed_chunks': vector_count,
            'total_tokens': sum(c.get('tokens', 0) for c in chunks),
            'embedding_model': self.embedding_model,
            'indexing_time': elapsed
        }
        
        print(f"Indexing complete in {elapsed:.2f}s")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total tokens: {stats['total_tokens']}")
        
        return stats
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        return_evidence: bool = True
    ) -> Dict:
        """
        Query the indexed documents.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            return_evidence: Include retrieved chunks in response
            
        Returns:
            Query results with retrieved chunks
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_single(question)
        
        # Retrieve chunks
        if self.use_hybrid:
            results = self.retriever.search(
                query=question,
                query_embedding=query_embedding,
                top_k=top_k
            )
        else:
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k
            )
        
        response = {
            'question': question,
            'retrieved_chunks': len(results),
            'top_k': top_k
        }
        
        if return_evidence:
            response['chunks'] = results
        
        return response
    
    def analyze_policy(
        self,
        text: str,
        url: str,
        company_name: str,
        dimensions: Optional[List[str]] = None,
        top_k: int = 10,
        llm_client = None
    ) -> Dict:
        """
        Full end-to-end analysis of a privacy policy.
        
        Args:
            text: Full policy text
            url: Policy URL
            company_name: Company name
            dimensions: List of dimension names to analyze (None = all)
            top_k: Chunks to retrieve per question
            llm_client: LLM client for generating answers (ChatGroq, OpenAI, etc.)
            
        Returns:
            Complete analysis results
        """
        print(f"\n{'='*60}")
        print(f"Analyzing Privacy Policy: {company_name}")
        print(f"{'='*60}\n")
        
        # Step 1: Index document
        index_stats = self.index_document(text, url)
        
        # Step 2: Get questions from template
        all_dimensions = self.template.get('analysis_dimensions', [])
        
        if dimensions:
            # Filter to requested dimensions
            all_dimensions = [d for d in all_dimensions if d['name'] in dimensions]
        
        # Flatten questions
        questions = []
        for dim in all_dimensions:
            for q in dim['questions']:
                questions.append({
                    'dimension': dim['name'],
                    'question': q
                })
        
        print(f"Analyzing {len(questions)} questions across {len(all_dimensions)} dimensions\n")
        
        # Step 3: Query and analyze
        analysis_results = []
        
        for i, item in enumerate(questions, 1):
            question = item['question']
            dimension = item['dimension']
            
            print(f"[{i}/{len(questions)}] {dimension}: {question[:60]}...")
            
            # Retrieve relevant chunks
            query_result = self.query(question, top_k=top_k, return_evidence=True)
            chunks = query_result.get('chunks', [])
            
            # Format context for LLM
            context = self._format_context(chunks)
            
            # Generate answer using LLM
            if llm_client:
                llm_response = self._call_llm(llm_client, question, context)
            else:
                # Fallback: simple extraction without LLM
                llm_response = {
                    'answer': 'LLM not configured. Retrieved relevant chunks only.',
                    'evidence': [
                        {
                            'chunk_id': str(c.get('chunk_id', '')),
                            'quote': c.get('text', '')[:200],
                            'relevance': 'unknown'
                        }
                        for c in chunks[:3]
                    ],
                    'confidence': 'low',
                    'coverage': 'unknown'
                }
            
            analysis_results.append({
                'dimension': dimension,
                'question': question,
                'response': llm_response,
                'retrieved_chunks': len(chunks)
            })
        
        # Step 4: Generate report
        print("\nGenerating report...")
        
        metadata = {
            **index_stats,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'top_k': top_k,
            'processing_time': index_stats.get('indexing_time', 0),
            'chunk_count': index_stats.get('total_chunks', 0)
        }
        
        report_md = self.reporter.generate_report(
            url=url,
            company_name=company_name,
            analysis_results=analysis_results,
            metadata=metadata
        )
        
        # Save report
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        safe_name = company_name.replace(' ', '_').replace('/', '_')
        report_path = output_dir / f"{safe_name}_analysis.md"
        
        self.reporter.save_report(report_md, str(report_path))
        
        print(f"\n{'='*60}")
        print(f"Analysis complete!")
        print(f"Report saved to: {report_path}")
        print(f"{'='*60}\n")
        
        return {
            'company': company_name,
            'url': url,
            'analysis_results': analysis_results,
            'report_path': str(report_path),
            'metadata': metadata
        }
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks as context for LLM"""
        context_template = self.template.get('context_template', '')
        
        formatted_chunks = []
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            formatted = context_template.format(
                chunk_id=chunk.get('chunk_id', ''),
                url=metadata.get('url', ''),
                header=metadata.get('header', 'N/A'),
                text=chunk.get('text', '')
            )
            formatted_chunks.append(formatted)
        
        return '\n\n'.join(formatted_chunks)
    
    def _call_llm(self, llm_client, question: str, context: str) -> Dict:
        """
        Call LLM to generate structured answer.
        
        Args:
            llm_client: LLM client (ChatGroq, ChatOpenAI, etc.)
            question: Question to answer
            context: Retrieved context chunks
            
        Returns:
            Structured response dictionary
        """
        try:
            # Build prompt
            system_prompt = self.template.get('system_prompt', '')
            prompt_template = self.template.get('prompt_template', '')
            schema = json.dumps(self.template.get('output_schema', {}), indent=2)
            
            user_prompt = prompt_template.format(
                context=context,
                question=question,
                schema=schema
            )
            
            # Call LLM
            from langchain.schema import SystemMessage, HumanMessage
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm_client.invoke(messages)
            response_text = response.content
            
            # Parse JSON response
            # Try to extract JSON from code blocks if present
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()
            elif '```' in response_text:
                json_start = response_text.find('```') + 3
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()
            
            parsed = json.loads(response_text)
            return parsed
            
        except Exception as e:
            print(f"  LLM error: {e}")
            return {
                'answer': 'Error generating answer',
                'evidence': [],
                'confidence': 'low',
                'coverage': 'none'
            }
    
    def clear_all(self):
        """Clear all cached data and indices"""
        self.vector_store.delete_all()
        self.embedder.clear_cache()
        print("All data cleared")
