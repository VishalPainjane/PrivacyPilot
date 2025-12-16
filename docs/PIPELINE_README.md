# PrivacyPilot v2.0 - Advanced RAG Pipeline

## Overview

PrivacyPilot is an advanced privacy policy analysis tool powered by hybrid Retrieval-Augmented Generation (RAG). It combines **token-aware chunking**, **dense vector search**, **BM25 lexical matching**, and **LLM reasoning** to provide comprehensive, evidence-based analysis of privacy policies.

## Key Features

### 🎯 Hybrid Retrieval System
- **Dense Vector Search**: Semantic similarity using sentence-transformers
- **BM25 Lexical Search**: Keyword-based retrieval for precise matching
- **Weighted Fusion**: Configurable α/β weights (default: 0.4 BM25, 0.6 vector)

### 📊 Token-Aware Chunking
- **Smart Boundaries**: Respects section headers and sentence boundaries
- **Configurable Size**: Default 512 tokens with 100-token overlap (~20%)
- **Header Preservation**: Keeps section context intact

### 💾 Intelligent Caching
- **SHA256-Based**: Deduplicated embedding storage
- **Persistent**: Survives sessions and reduces costs
- **Batch Processing**: Optimized for large documents

### 📝 Structured Output
- **Markdown-First**: Human-readable reports
- **Evidence Tracking**: Every claim linked to source chunks
- **Confidence Scores**: High/medium/low confidence levels
- **Coverage Assessment**: Complete/partial/none coverage

### 🧠 LLM Integration
- **ChatGroq**: llama-3.3-70b-versatile (fast, accurate)
- **Structured Prompts**: JSON schema enforcement
- **Few-Shot Examples**: Improved consistency
- **Citation Requirements**: All answers must cite chunk IDs

## Architecture

```
┌─────────────────┐
│  Privacy Policy │
│   (Raw Text)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Token-Aware    │ ← 512 tokens, 100 overlap
│    Chunking     │ ← Header-aware splitting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding Gen  │ ← sentence-transformers/all-MiniLM-L6-v2
│  + SHA256 Cache │ ← Batch processing (32)
└────────┬────────┘
         │
         ├────────────┬────────────┐
         ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ Chroma  │  │  BM25   │  │  Both   │
    │ (Dense) │  │(Lexical)│  │(Hybrid) │
    └────┬────┘  └────┬────┘  └────┬────┘
         │            │            │
         └────────────┴────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  Retrieval   │ ← Top-K chunks
              │  (α BM25 +   │ ← Score fusion
              │   β Vector)  │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │  LLM Query   │ ← ChatGroq
              │  (Groq API)  │ ← JSON output
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │   Markdown   │
              │    Report    │
              └──────────────┘
```

## Installation

### 1. Clone Repository
```bash
git clone <repo-url>
cd PrivacyPilot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## Quick Start

### Run Test Suite
```bash
python test_pipeline.py
```

### Analyze a Privacy Policy
```python
from pipeline import PrivacyPolicyPipeline
from langchain_groq import ChatGroq
import os

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize pipeline
pipeline = PrivacyPolicyPipeline(
    chunk_size=512,
    overlap=100,
    use_hybrid=True,
    alpha=0.4,  # BM25 weight
    beta=0.6    # Vector weight
)

# Load your privacy policy
with open("policy.txt", "r") as f:
    policy_text = f.read()

# Run analysis
result = pipeline.analyze_policy(
    text=policy_text,
    url="https://company.com/privacy",
    company_name="Company Name",
    llm_client=llm
)

print(f"Report saved to: {result['report_path']}")
```

### Custom Queries
```python
# Index a document
pipeline.index_document(
    text=policy_text,
    url="https://company.com/privacy"
)

# Query specific questions
result = pipeline.query(
    question="How long is my data retained?",
    top_k=5
)

# Examine retrieved chunks
for chunk in result['chunks']:
    print(f"Score: {chunk['hybrid_score']:.3f}")
    print(f"Text: {chunk['text'][:200]}...")
```

## Pipeline Modules

### `pipeline/chunker.py`
- **TokenAwareChunker**: Token-based chunking with header awareness
- **chunk_privacy_policy()**: Convenience function

**Configuration:**
- `chunk_tokens`: Target chunk size (default: 512)
- `overlap_tokens`: Overlap between chunks (default: 100)
- `preserve_headers`: Keep headers with content (default: True)

### `pipeline/embedder.py`
- **CachedEmbedder**: Embedding generator with SHA256 caching
- **embed_privacy_chunks()**: Convenience function

**Configuration:**
- `model_name`: HuggingFace model (default: all-MiniLM-L6-v2)
- `cache_dir`: Cache directory (default: .embedding_cache)
- `batch_size`: Batch size for encoding (default: 32)

### `pipeline/indexer.py`
- **ChromaIndexer**: Vector store (local, fast)
- **BM25Retriever**: Lexical search (TF-IDF fallback)
- **HybridRetriever**: Combines both with weighted fusion

**Configuration:**
- `alpha`: BM25 weight (default: 0.4)
- `beta`: Vector weight (default: 0.6)
- `top_k`: Results to return (default: 10)

### `pipeline/reporter.py`
- **MarkdownReporter**: Generate structured reports
- **generate_privacy_report()**: Convenience function

**Features:**
- Executive summary
- Dimensional analysis
- Evidence tables
- Coverage assessment

### `pipeline/rag_pipeline.py`
- **PrivacyPolicyPipeline**: Main orchestrator
- End-to-end analysis workflow

## Prompt Engineering

The system uses structured prompts defined in `pipeline/prompt_template.json`:

### System Prompt
- Expert privacy analyst persona
- Evidence-based analysis requirements
- Citation enforcement

### Output Schema (JSON)
```json
{
  "answer": "string",
  "evidence": [
    {
      "chunk_id": "string",
      "quote": "string",
      "relevance": "high|medium|low"
    }
  ],
  "confidence": "high|medium|low",
  "coverage": "complete|partial|none"
}
```

### Analysis Dimensions
1. **Data Collection**: What, how, when
2. **Data Usage**: Purposes, profiling, marketing
3. **Data Sharing**: Third parties, transfers
4. **Data Retention**: Duration, deletion
5. **User Rights**: Access, deletion, portability
6. **Security**: Encryption, breach notification
7. **Children's Privacy**: Age limits, consent
8. **Policy Changes**: Notification procedures
9. **Legal Basis (GDPR)**: Consent, legitimate interest
10. **Contact & Complaints**: Support channels

## Configuration

### Chunking Strategy
```python
chunker = TokenAwareChunker(
    chunk_tokens=512,      # Adjust based on policy length
    overlap_tokens=100,    # 20% overlap recommended
    preserve_headers=True  # Keep section context
)
```

### Embedding Model
```python
embedder = CachedEmbedder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast, accurate
    # Alternatives:
    # - "all-mpnet-base-v2" (higher quality, slower)
    # - "paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
)
```

### Hybrid Weights
```python
retriever = HybridRetriever(
    alpha=0.4,  # BM25 weight (keyword matching)
    beta=0.6    # Vector weight (semantic similarity)
)
# Adjust based on query type:
# - More keywords → increase alpha
# - More semantic → increase beta
```

## Performance

### Benchmarks (GitHub Terms of Service - 182KB)
- **Chunking**: ~0.15s (13 chunks, 344 tokens)
- **Embedding** (cached): ~0.01s
- **Embedding** (uncached): ~0.2s (batch of 13)
- **Indexing**: ~0.05s
- **Retrieval**: ~0.02s per query
- **LLM Call**: ~2-5s per question

**Total Analysis** (10 dimensions, 40 questions): ~3-5 minutes

## Troubleshooting

### "NotImplementedError: Event loop on Windows"
✅ **Fixed** - Pipeline uses thread pool execution for async operations

### "Failed to send telemetry event"
ℹ️ **Harmless** - Chroma telemetry warnings, doesn't affect functionality

### "LLM error: Extra data"
⚠️ **LLM Output** - Model sometimes returns explanatory text after JSON. Parser attempts extraction.

### "Permission denied: chroma.sqlite3"
🔒 **Windows File Lock** - Close ChromaDB connections before cleanup:
```python
pipeline.vector_store = None
```

## Project Structure

```
PrivacyPilot/
├── pipeline/
│   ├── __init__.py           # Package exports
│   ├── chunker.py            # Token-aware chunking
│   ├── embedder.py           # Cached embedding generation
│   ├── indexer.py            # Vector store & BM25
│   ├── reporter.py           # Markdown report generation
│   ├── rag_pipeline.py       # Main orchestrator
│   └── prompt_template.json  # Prompt engineering
├── scrape/
│   ├── scrape.py             # Web scraping (crawl4ai)
│   └── extract_link.py       # Google search
├── reports/                  # Generated reports
├── .embedding_cache/         # Embedding cache (SHA256)
├── .chroma_db/              # Vector database
├── test_pipeline.py         # Test suite
├── main2.py                 # Legacy pipeline
├── app.py                   # Streamlit app
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Contributing

### Adding New Analysis Dimensions
1. Edit `pipeline/prompt_template.json`
2. Add dimension to `analysis_dimensions` array
3. Define questions for the dimension

### Improving Prompt Engineering
- Modify `system_prompt` for persona/requirements
- Add `few_shot_examples` for consistency
- Update `output_schema` for new fields

### Custom Retrievers
Implement retriever interface:
```python
class CustomRetriever:
    def search(self, query, query_embedding, top_k):
        # Return list of dicts with 'chunk_id', 'text', 'score'
        pass
```

## Citation

If you use PrivacyPilot in research:
```bibtex
@software{privacypilot2024,
  title = {PrivacyPilot: Advanced RAG-Based Privacy Policy Analysis},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/PrivacyPilot}
}
```

## License

MIT License - See LICENSE file

## Acknowledgments

- **crawl4ai**: Web scraping framework
- **sentence-transformers**: Embedding models
- **ChromaDB**: Vector database
- **Groq**: LLM inference
- **LangChain**: LLM orchestration

---

**Version**: 2.0  
**Last Updated**: 2024  
**Status**: ✅ Production Ready
