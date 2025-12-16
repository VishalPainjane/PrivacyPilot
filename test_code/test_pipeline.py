"""
Test script for the new RAG pipeline.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pipeline import PrivacyPolicyPipeline

# Load environment
load_dotenv()

def test_chunking():
    """Test the token-aware chunker"""
    print("\n" + "="*60)
    print("TEST 1: Token-Aware Chunking")
    print("="*60)
    
    from pipeline.chunker import TokenAwareChunker
    
    sample_text = """
PRIVACY POLICY

1. DATA COLLECTION

We collect personal information including your name, email address, and IP address.

1.1 Account Information
When you create an account, we collect your username and password.

1.2 Usage Data
We automatically collect information about how you use our services.

2. DATA USAGE

We use your data for the following purposes:
- To provide and maintain our services
- To improve user experience
- For analytics and research

3. DATA SHARING

We may share your information with:
- Service providers who assist in operations
- Law enforcement when required by law
- Business partners with your consent
    """
    
    chunker = TokenAwareChunker(chunk_tokens=100, overlap_tokens=20)
    chunks = chunker.chunk_document(sample_text, url="https://example.com/privacy")
    
    print(f"\nCreated {len(chunks)} chunks from sample policy")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i}:")
        print(f"  Tokens: {chunk['tokens']}")
        print(f"  Header: {chunk.get('metadata', {}).get('header', 'N/A')}")
        print(f"  Text: {chunk['text'][:100]}...")
    
    return len(chunks) > 0

def test_embedding():
    """Test the cached embedder"""
    print("\n" + "="*60)
    print("TEST 2: Cached Embedding Generation")
    print("="*60)
    
    from pipeline.embedder import CachedEmbedder
    
    embedder = CachedEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=".test_cache"
    )
    
    texts = [
        "We collect your email address and name.",
        "Your data is encrypted in transit and at rest.",
        "You can request deletion of your account at any time."
    ]
    
    embeddings = embedder.embed_batch(texts, use_cache=True)
    
    print(f"\nGenerated embeddings: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
    
    stats = embedder.cache_stats()
    print(f"Cache stats: {stats}")
    
    # Test cache hit
    print("\nTesting cache hit...")
    embeddings2 = embedder.embed_batch(texts, use_cache=True)
    print(f"Second batch (should be cached): {embeddings2.shape}")
    
    return embeddings.shape[0] == len(texts)

def test_indexing():
    """Test vector store indexing"""
    print("\n" + "="*60)
    print("TEST 3: Vector Store Indexing")
    print("="*60)
    
    from pipeline.chunker import chunk_privacy_policy
    from pipeline.embedder import CachedEmbedder
    from pipeline.indexer import ChromaIndexer
    
    # Sample document
    text = "We collect your personal information. We use encryption for security. You can delete your account."
    
    # Chunk and embed
    chunks = chunk_privacy_policy(text, url="https://example.com/privacy", chunk_size=50, overlap=10)
    
    embedder = CachedEmbedder(cache_dir=".test_cache")
    chunks = embedder.embed_chunks(chunks)
    
    # Index
    indexer = ChromaIndexer(persist_directory=".test_chroma")
    count = indexer.add_chunks(chunks)
    
    print(f"\nIndexed {count} chunks")
    print(f"Total documents in collection: {indexer.count()}")
    
    # Test search
    query_emb = embedder.embed_single("How do you protect my data?")
    results = indexer.search(query_emb, top_k=2)
    
    print(f"\nSearch results for 'How do you protect my data?':")
    for i, result in enumerate(results):
        print(f"\n  Result {i+1}:")
        print(f"    Score: {result['score']:.3f}")
        print(f"    Text: {result['text'][:100]}...")
    
    return len(results) > 0

def test_full_pipeline():
    """Test the complete RAG pipeline"""
    print("\n" + "="*60)
    print("TEST 4: Complete RAG Pipeline")
    print("="*60)
    
    # Sample privacy policy
    sample_policy = """
PRIVACY POLICY - Example Corp

INTRODUCTION
This privacy policy explains how Example Corp ("we", "us", or "our") collects, uses, and protects your personal information.

DATA COLLECTION
We collect the following types of information:
- Personal identifiers: name, email address, phone number
- Account credentials: username and password
- Usage information: pages visited, features used, time spent
- Device information: IP address, browser type, operating system

DATA USAGE
We use your information for:
1. Providing and maintaining our services
2. Personalizing your experience
3. Analyzing usage patterns to improve our platform
4. Communicating updates and promotional offers
5. Complying with legal obligations

DATA SHARING
We may share your information with:
- Service providers who help operate our platform
- Analytics partners for usage analysis
- Law enforcement when legally required
- Third-party advertisers (with your consent)

DATA RETENTION
We retain your data:
- Account information: while your account is active
- Usage logs: for 90 days
- Marketing data: until you opt out
- Legal compliance data: as required by law

USER RIGHTS
You have the right to:
- Access your personal data
- Request correction of inaccurate data
- Delete your account and associated data
- Opt out of marketing communications
- Export your data in machine-readable format

SECURITY
We implement industry-standard security measures including:
- Encryption of data in transit (TLS 1.3)
- Encryption of data at rest (AES-256)
- Regular security audits
- Access controls and authentication

CONTACT
For privacy inquiries, contact us at privacy@example.com
    """
    
    # Initialize pipeline
    pipeline = PrivacyPolicyPipeline(
        chunk_size=256,
        overlap=50,
        use_hybrid=True
    )
    
    # Index the document
    stats = pipeline.index_document(
        text=sample_policy,
        url="https://example.com/privacy"
    )
    
    print(f"\nIndexing stats: {stats}")
    
    # Test queries
    test_questions = [
        "What personal data is collected?",
        "How long is my data retained?",
        "Can I delete my account?"
    ]
    
    print("\nTesting queries:")
    for question in test_questions:
        result = pipeline.query(question, top_k=3)
        print(f"\nQ: {question}")
        print(f"Retrieved {result['retrieved_chunks']} chunks")
        
        if result.get('chunks'):
            top_chunk = result['chunks'][0]
            print(f"Top result (score: {top_chunk.get('hybrid_score', top_chunk.get('score', 0)):.3f}):")
            print(f"  {top_chunk['text'][:150]}...")
    
    return True

def test_with_llm():
    """Test full analysis with LLM"""
    print("\n" + "="*60)
    print("TEST 5: Complete Analysis with LLM")
    print("="*60)
    
    if not os.getenv("GROQ_API_KEY"):
        print("Skipping LLM test - GROQ_API_KEY not set")
        return False
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    
    # Sample policy
    sample_policy = """
PRIVACY NOTICE

1. INFORMATION WE COLLECT
We collect name, email, and payment information when you register.

2. HOW WE USE YOUR DATA
Your data is used to process orders and improve our services.

3. DATA RETENTION
We keep your data for 3 years after account closure.

4. YOUR RIGHTS
You can request data deletion by emailing support@company.com.
    """
    
    # Initialize pipeline
    pipeline = PrivacyPolicyPipeline(chunk_size=128, overlap=25)
    
    # Run full analysis (subset of dimensions)
    result = pipeline.analyze_policy(
        text=sample_policy,
        url="https://company.com/privacy",
        company_name="Test Company",
        dimensions=["Data Collection", "Data Retention", "User Rights"],
        top_k=5,
        llm_client=llm
    )
    
    print(f"\nAnalysis complete!")
    print(f"Report saved to: {result['report_path']}")
    print(f"Analyzed {len(result['analysis_results'])} questions")
    
    # Show sample result
    if result['analysis_results']:
        sample = result['analysis_results'][0]
        print(f"\nSample analysis:")
        print(f"Q: {sample['question']}")
        print(f"A: {sample['response'].get('answer', 'N/A')[:200]}...")
        print(f"Confidence: {sample['response'].get('confidence', 'N/A')}")
    
    return True

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PRIVACY PILOT RAG PIPELINE - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Chunking", test_chunking),
        ("Embedding", test_embedding),
        ("Indexing", test_indexing),
        ("Full Pipeline", test_full_pipeline),
        ("LLM Analysis", test_with_llm)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            results[name] = "ERROR"
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASS" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    # Cleanup
    print("\nCleaning up test data...")
    import shutil
    import time
    
    # Close ChromaDB connections first
    try:
        import gc
        gc.collect()  # Force garbage collection to close file handles
        time.sleep(0.5)  # Give OS time to release locks
    except:
        pass
    
    for path in [".test_cache", ".test_chroma"]:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"Removed {path}")
            except PermissionError:
                print(f"⚠️  Could not remove {path} (files in use). Please delete manually.")
    print("Cleanup complete!")

if __name__ == "__main__":
    main()
