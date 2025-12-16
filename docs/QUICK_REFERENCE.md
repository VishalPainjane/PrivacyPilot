# Quick Reference Guide - PrivacyPilot v2.0

## 🚀 Quick Start (3 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

### 3. Run Test
```bash
python test_pipeline.py
```

### 4. Launch App
```bash
streamlit run app_v2.py
```

---

## 📖 Common Usage Patterns

### Pattern 1: Analyze Single Policy
```python
from pipeline import PrivacyPolicyPipeline
from langchain_groq import ChatGroq
import os

# Setup
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
pipeline = PrivacyPolicyPipeline()

# Load policy
with open("policy.txt") as f:
    text = f.read()

# Analyze
result = pipeline.analyze_policy(
    text=text,
    url="https://company.com/privacy",
    company_name="Company",
    llm_client=llm
)

print(f"Report: {result['report_path']}")
```

### Pattern 2: Custom Query
```python
# Index once
pipeline.index_document(text, url)

# Query multiple times
questions = [
    "How long is my data kept?",
    "Can I delete my account?",
    "Is my data encrypted?"
]

for q in questions:
    result = pipeline.query(q, top_k=5)
    print(f"\nQ: {q}")
    print(f"Top result: {result['chunks'][0]['text'][:200]}")
```

### Pattern 3: Compare Retrievers
```python
# Vector only
pipeline_vector = PrivacyPolicyPipeline(use_hybrid=False)

# Hybrid (BM25 + Vector)
pipeline_hybrid = PrivacyPolicyPipeline(
    use_hybrid=True,
    alpha=0.4,  # BM25
    beta=0.6    # Vector
)

# Compare
query = "data retention period"
r1 = pipeline_vector.query(query)
r2 = pipeline_hybrid.query(query)

print(f"Vector: {r1['chunks'][0]['text'][:100]}")
print(f"Hybrid: {r2['chunks'][0]['text'][:100]}")
```

---

## ⚙️ Configuration Cheat Sheet

### Chunking
```python
from pipeline.chunker import TokenAwareChunker

chunker = TokenAwareChunker(
    chunk_tokens=512,      # Token size (256-1024)
    overlap_tokens=100,    # Overlap (50-200, ~20%)
    preserve_headers=True  # Keep headers with content
)
```

**Recommendations:**
- Short policies (<5K tokens): 256 tokens, 50 overlap
- Medium policies (5-20K tokens): 512 tokens, 100 overlap
- Long policies (>20K tokens): 768 tokens, 150 overlap

### Embeddings
```python
from pipeline.embedder import CachedEmbedder

embedder = CachedEmbedder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast
    # model_name="all-mpnet-base-v2",  # Better quality
    cache_dir=".embedding_cache",
    batch_size=32
)
```

**Model Comparison:**
| Model | Dim | Speed | Quality |
|-------|-----|-------|---------|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| all-mpnet-base-v2 | 768 | Medium | Better |
| paraphrase-multilingual | 384 | Fast | Multilingual |

### Retrieval Weights
```python
# BM25-heavy (keyword matching)
α = 0.7, β = 0.3  # For technical terms, specific dates

# Balanced
α = 0.4, β = 0.6  # Default, most cases

# Vector-heavy (semantic)
α = 0.2, β = 0.8  # For conceptual questions
```

---

## 🔧 Troubleshooting

### Issue: Slow Embedding
**Solution:** Check cache
```python
stats = embedder.cache_stats()
print(stats)  # Should show cached_embeddings > 0

# Clear cache if corrupted
embedder.clear_cache()
```

### Issue: Poor Retrieval
**Solution:** Adjust top-k and weights
```python
# Try different top-k
for k in [5, 10, 15, 20]:
    result = pipeline.query(question, top_k=k)
    print(f"Top-{k}: {len(result['chunks'])} chunks")

# Try different weights
pipeline.retriever.alpha = 0.6  # More BM25
pipeline.retriever.beta = 0.4   # Less vector
```

### Issue: LLM JSON Errors
**Solution:** Already handled in parser
```python
# Parser extracts JSON from:
# ```json
# {...}
# ```
# Or plain {...}
```

### Issue: Out of Memory
**Solution:** Reduce batch size
```python
embedder = CachedEmbedder(batch_size=8)  # Default is 32
```

---

## 📊 Pipeline Components at a Glance

```
INPUT (Policy Text)
    ↓
┌───────────────────┐
│ TokenAwareChunker │ → 512 tokens, 100 overlap, header-aware
└────────┬──────────┘
         ↓
┌───────────────────┐
│  CachedEmbedder   │ → all-MiniLM-L6-v2, SHA256 cache
└────────┬──────────┘
         ↓
    ┌────┴────┐
    ↓         ↓
┌──────┐  ┌──────┐
│Chroma│  │ BM25 │
│Vector│  │Lexical│
└───┬──┘  └──┬───┘
    └────┬────┘
         ↓
┌───────────────────┐
│ HybridRetriever   │ → 0.4×BM25 + 0.6×Vector
└────────┬──────────┘
         ↓
┌───────────────────┐
│   LLM (ChatGroq)  │ → llama-3.3-70b-versatile
└────────┬──────────┘
         ↓
┌───────────────────┐
│ MarkdownReporter  │ → Report with evidence
└───────────────────┘
```

---

## 🎯 API Reference (Essential Functions)

### PrivacyPolicyPipeline
```python
pipeline = PrivacyPolicyPipeline(
    chunk_size=512,
    overlap=100,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_hybrid=True,
    alpha=0.4,
    beta=0.6,
    cache_dir=".embedding_cache",
    chroma_dir=".chroma_db",
    template_path="pipeline/prompt_template.json"
)

# Main methods
pipeline.index_document(text, url) → dict (stats)
pipeline.query(question, top_k=10) → dict (results)
pipeline.analyze_policy(text, url, company, dimensions, llm_client) → dict
pipeline.clear_all() → None
```

### Convenience Functions
```python
from pipeline import (
    chunk_privacy_policy,
    embed_privacy_chunks,
    generate_privacy_report
)

# Quick chunking
chunks = chunk_privacy_policy(text, url, chunk_size=512, overlap=100)

# Quick embedding
chunks_with_emb = embed_privacy_chunks(chunks)

# Quick report
report_path = generate_privacy_report(
    url=url,
    company_name=company,
    analysis_results=results,
    save_pdf=True
)
```

---

## 📈 Performance Tips

### 1. Batch Processing
```python
# Process multiple policies
policies = [...]

for policy in policies:
    # Embeddings are cached across runs
    result = pipeline.analyze_policy(
        text=policy['text'],
        url=policy['url'],
        company_name=policy['company'],
        llm_client=llm
    )
```

### 2. Parallel Queries
```python
from concurrent.futures import ThreadPoolExecutor

questions = [...]

with ThreadPoolExecutor(max_workers=5) as executor:
    results = executor.map(
        lambda q: pipeline.query(q, top_k=5),
        questions
    )
```

### 3. Dimension Filtering
```python
# Only analyze specific dimensions
result = pipeline.analyze_policy(
    text=text,
    url=url,
    company_name=company,
    dimensions=["Data Collection", "User Rights"],  # Faster!
    llm_client=llm
)
```

---

## 🎨 Customization Examples

### Custom Analysis Dimensions
Edit `pipeline/prompt_template.json`:
```json
{
  "analysis_dimensions": [
    {
      "name": "Custom Dimension",
      "questions": [
        "Your custom question 1?",
        "Your custom question 2?"
      ]
    }
  ]
}
```

### Custom Retriever
```python
class CustomRetriever:
    def search(self, query, query_embedding, top_k):
        # Your logic here
        return [
            {
                'chunk_id': '...',
                'text': '...',
                'score': 0.95
            }
        ]

# Use it
pipeline.retriever = CustomRetriever()
```

### Custom Report Format
```python
from pipeline.reporter import MarkdownReporter

class CustomReporter(MarkdownReporter):
    def _generate_header(self, company_name, url, timestamp, metadata):
        # Custom header format
        return f"# {company_name} Analysis\n..."

pipeline.reporter = CustomReporter()
```

---

## 🔍 Debugging Commands

```python
# Check chunk count
chunker = TokenAwareChunker()
chunks = chunker.chunk_document(text, url)
print(f"Chunks: {len(chunks)}")

# Check embeddings
embedder = CachedEmbedder()
stats = embedder.cache_stats()
print(stats)

# Check vector store
print(f"Indexed docs: {pipeline.vector_store.count()}")

# Test retrieval
query_emb = embedder.embed_single("test query")
results = pipeline.vector_store.search(query_emb, top_k=3)
for r in results:
    print(f"Score: {r['score']:.3f}, Text: {r['text'][:50]}")
```

---

## 📦 File Structure Quick Reference

```
PrivacyPilot/
├── pipeline/              # Core RAG system
│   ├── chunker.py        # Token-aware splitting
│   ├── embedder.py       # Cached embeddings
│   ├── indexer.py        # Chroma + BM25 + Hybrid
│   ├── reporter.py       # Markdown reports
│   ├── rag_pipeline.py   # Main orchestrator
│   └── prompt_template.json  # Prompts + dimensions
├── test_pipeline.py      # Test suite
├── app_v2.py            # Streamlit UI
├── PIPELINE_README.md   # Full documentation
└── IMPLEMENTATION_SUMMARY.md  # Implementation details
```

---

## ✅ Quick Checklist

**Before Running:**
- [ ] `pip install -r requirements.txt`
- [ ] Create `.env` with `GROQ_API_KEY`
- [ ] Run `python test_pipeline.py` (should pass 5/5)

**For Production:**
- [ ] Set appropriate chunk size (512 recommended)
- [ ] Configure hybrid weights (0.4/0.6 default)
- [ ] Enable caching (default enabled)
- [ ] Filter dimensions if needed (speeds up analysis)

**Monitoring:**
- [ ] Check cache hit rate: `embedder.cache_stats()`
- [ ] Monitor LLM costs: ~40 calls per full analysis
- [ ] Track retrieval quality: Review evidence chunks

---

**For full details, see:** `PIPELINE_README.md`  
**For implementation notes, see:** `IMPLEMENTATION_SUMMARY.md`
