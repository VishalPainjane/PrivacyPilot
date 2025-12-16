"""
Integration example: Using the new RAG pipeline with existing Streamlit app.

This shows how to migrate from the old pipeline (main2.py) to the new RAG pipeline
while keeping the Streamlit interface intact.
"""
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pipeline import PrivacyPolicyPipeline
from pipeline.gdpr_scorer import GDPRScorer
from scrape.scrape_v2 import scrape
from scrape.extract_link import get_first_google_result
import time

# Load environment
load_dotenv()

st.set_page_config(
    page_title="PrivacyPilot v2.0",
    page_icon="🔒",
    layout="wide"
)

st.title("🔒 PrivacyPilot v2.0 - Advanced Privacy Analysis")
st.markdown("**Powered by Hybrid RAG** | Token-aware chunking + BM25 + Vector search")

# Initialize session state
if 'pipeline' not in st.session_state:
    with st.spinner("Initializing RAG pipeline..."):
        st.session_state.pipeline = PrivacyPolicyPipeline(
            chunk_size=512,
            overlap=100,
            use_hybrid=True,
            alpha=0.4,
            beta=0.6
        )
        st.success("Pipeline initialized!")

if 'gdpr_scorer' not in st.session_state:
    st.session_state.gdpr_scorer = GDPRScorer()

if 'llm' not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.session_state.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0
        )
    else:
        st.warning("GROQ_API_KEY not set. Analysis will use retrieval only.")
        st.session_state.llm = None

# Initialize processing state
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Input method
input_method = st.sidebar.radio(
    "Input Method",
    ["URL", "Company Name", "Direct Text"]
)

# Advanced settings
with st.sidebar.expander("🔧 Advanced Settings"):
    chunk_size = st.slider("Chunk Size (tokens)", 256, 1024, 512, 64)
    overlap = st.slider("Overlap (tokens)", 50, 200, 100, 25)
    top_k = st.slider("Top-K Retrieval", 3, 20, 10, 1)
    
    use_hybrid = st.checkbox("Hybrid Retrieval", value=True)
    if use_hybrid:
        alpha = st.slider("BM25 Weight (α)", 0.0, 1.0, 0.4, 0.1)
        beta = st.slider("Vector Weight (β)", 0.0, 1.0, 0.6, 0.1)
        st.info(f"Hybrid: {alpha:.1f} × BM25 + {beta:.1f} × Vector")
    
    # Dimension selection
    st.markdown("**Analysis Dimensions**")
    dimensions = [
        "Data Collection",
        "Data Usage",
        "Data Sharing",
        "Data Retention",
        "User Rights",
        "Security",
        "Children's Privacy",
        "Policy Changes",
        "Legal Basis (GDPR)",
        "Contact & Complaints"
    ]
    selected_dims = st.multiselect(
        "Select dimensions to analyze",
        dimensions,
        default=dimensions[:5]  # Default to first 5
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📥 Input Privacy Policy")
    
    if input_method == "URL":
        url = st.text_input("Privacy Policy URL", placeholder="https://example.com/privacy", disabled=st.session_state.is_processing)
        
        btn_disabled = st.session_state.is_processing or not url
        if st.button("🔍 Scrape & Analyze", key="url_btn", disabled=btn_disabled, type="primary"):
            if url:
                st.session_state.is_processing = True
                status_placeholder = st.empty()
                
                try:
                    # Step 1: Validate URL
                    if not (url.startswith('http://') or url.startswith('https://')):
                        url = 'https://' + url
                    
                    status_placeholder.info(f"🌐 Connecting to {url}...")
                    time.sleep(0.5)
                    
                    # Step 2: Scrape
                    status_placeholder.info(f"📥 Scraping content (this may take 5-10 seconds)...")
                    policy_data = scrape(url)
                    policy_text = policy_data.get('text', '')
                    
                    # Step 3: Validate
                    if policy_text and len(policy_text.strip()) > 100:
                        st.session_state.policy_text = policy_text
                        st.session_state.policy_url = url
                        st.session_state.company_name = policy_data.get('title', 'Unknown Company')
                        st.session_state.policy_sources = policy_data.get('sources', [])
                        st.session_state.page_contents = policy_data.get('page_contents', {})
                        
                        status_placeholder.success(f"✅ Scraped {len(policy_text):,} characters from {len(st.session_state.policy_sources)} pages")
                        
                        # Show sources
                        if st.session_state.policy_sources:
                            with st.expander("📚 Sources crawled"):
                                for src in st.session_state.policy_sources:
                                    st.markdown(f"- [{src['title'][:60]}...]({src['url']}) - {src['length']:,} chars")
                        
                        time.sleep(1.5)
                    else:
                        status_placeholder.error(f"❌ Scraped content too short ({len(policy_text)} chars)")
                        status_placeholder.info("💡 The page might be behind authentication, use JavaScript heavily, or have anti-bot protection")
                        
                except Exception as e:
                    status_placeholder.error(f"❌ Error: {str(e)}")
                    with st.expander("🔍 Show detailed error"):
                        import traceback
                        st.code(traceback.format_exc())
                finally:
                    st.session_state.is_processing = False
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.warning("⚠️ Please enter a URL")
    
    elif input_method == "Company Name":
        company = st.text_input("Company Name", placeholder="Google, Facebook, etc.", disabled=st.session_state.is_processing)
        
        btn_disabled = st.session_state.is_processing or not company
        if st.button("🔍 Search & Analyze", key="company_btn", disabled=btn_disabled, type="primary"):
            if company:
                st.session_state.is_processing = True
                status_placeholder = st.empty()
                
                try:
                    # Step 1: Search Google
                    status_placeholder.info(f"🔎 Searching for '{company} privacy policy'...")
                    query = f"{company} privacy policy"
                    url = get_first_google_result(query)
                    
                    # Fallback: Try to construct a likely privacy policy URL
                    if not url:
                        status_placeholder.warning(f"⚠️ Search failed. Trying common privacy policy URLs...")
                        
                        # Try common privacy policy URL patterns
                        domain = company.lower().replace(' ', '')
                        common_urls = [
                            f"https://www.{domain}.com/privacy",
                            f"https://www.{domain}.com/privacy-policy",
                            f"https://{domain}.com/legal/privacy",
                            f"https://www.{domain}.com/policies/privacy",
                        ]
                        
                        # Try each URL pattern
                        for test_url in common_urls:
                            try:
                                import requests
                                response = requests.head(test_url, timeout=5, allow_redirects=True)
                                if response.status_code == 200:
                                    url = test_url
                                    status_placeholder.success(f"✅ Found likely URL: {url}")
                                    break
                            except:
                                continue
                    
                    if url:
                        if not url.startswith('http'):
                            url = 'https://' + url
                            
                        status_placeholder.success(f"✅ Found URL: {url}")
                        time.sleep(1)
                        
                        # Step 2: Scrape the URL
                        status_placeholder.info(f"📥 Scraping privacy policy (this may take 5-10 seconds)...")
                        policy_data = scrape(url)
                        policy_text = policy_data.get('text', '')
                        
                        # Step 3: Validate results
                        if policy_text and len(policy_text.strip()) > 100:
                            st.session_state.policy_text = policy_text
                            st.session_state.policy_url = url
                            st.session_state.company_name = company
                            status_placeholder.success(f"✅ Successfully scraped {len(policy_text):,} characters!")
                            time.sleep(1.5)
                        else:
                            status_placeholder.error(f"❌ Scraped content too short ({len(policy_text)} chars)")
                            status_placeholder.info("💡 The page might require authentication or have anti-bot protection")
                    else:
                        status_placeholder.error(f"❌ Could not find privacy policy for '{company}'")
                        status_placeholder.info("💡 Try entering the exact privacy policy URL using the 'URL' option above")
                        
                except Exception as e:
                    status_placeholder.error(f"❌ Error: {str(e)}")
                    with st.expander("🔍 Show detailed error"):
                        import traceback
                        st.code(traceback.format_exc())
                finally:
                    st.session_state.is_processing = False
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.warning("⚠️ Please enter a company name")
    
    else:  # Direct Text
        policy_text = st.text_area(
            "Paste Privacy Policy Text",
            height=200,
            placeholder="Paste the full privacy policy text here...",
            disabled=st.session_state.is_processing
        )
        company_name = st.text_input("Company Name (optional)", disabled=st.session_state.is_processing)
        
        btn_disabled = st.session_state.is_processing or not policy_text
        if st.button("📝 Load Text", key="text_btn", disabled=btn_disabled, type="primary"):
            if policy_text:
                st.session_state.policy_text = policy_text
                st.session_state.policy_url = "Direct Input"
                st.session_state.company_name = company_name or "Direct Input"
                st.success(f"✅ Policy loaded ({len(policy_text):,} characters)")
            else:
                st.warning("⚠️ Please paste policy text")

with col2:
    st.header("📊 Stats")
    
    if 'policy_text' in st.session_state:
        text = st.session_state.policy_text
        st.metric("Characters", f"{len(text):,}")
        st.metric("Words", f"{len(text.split()):,}")
        
        # Estimate chunks
        from pipeline.chunker import TokenAwareChunker
        chunker = TokenAwareChunker(chunk_tokens=chunk_size, overlap_tokens=overlap)
        token_count = chunker.count_tokens(text)
        est_chunks = max(1, token_count // (chunk_size - overlap))
        
        st.metric("Tokens (est.)", f"{token_count:,}")
        st.metric("Chunks (est.)", est_chunks)
        
        st.info(f"**Source:** {st.session_state.get('company_name', 'Unknown')}")
    else:
        st.info("Load a policy to see stats")

# Analysis section
st.markdown("---")

if 'policy_text' in st.session_state:
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 GDPR Compliance Score", "📋 Full Analysis", "❓ Custom Query", "📄 Report"])
    
    with tab1:
        st.subheader("🎯 GDPR Compliance Scoring")
        st.markdown("*Simple, clear scoring against 7 core GDPR principles*")
        
        btn_disabled = st.session_state.is_processing
        if st.button("📊 Score GDPR Compliance", type="primary", disabled=btn_disabled, key="gdpr_score_btn"):
            st.session_state.is_processing = True
            
            try:
                with st.spinner("🔍 Analyzing against GDPR principles..."):
                    policy_text = st.session_state.policy_text
                    
                    # Get some context chunks first
                    if st.session_state.pipeline.vector_store.count() == 0:
                        st.session_state.pipeline.index_document(
                            text=policy_text,
                            url=st.session_state.policy_url
                        )
                    
                    # Retrieve relevant chunks for evidence
                    gdpr_query = "data protection rights security transparency consent purpose"
                    chunks_result = st.session_state.pipeline.query(
                        question=gdpr_query,
                        top_k=20,
                        return_evidence=True
                    )
                    
                    # Score the policy
                    gdpr_results = st.session_state.gdpr_scorer.score_policy(
                        text=policy_text,
                        retrieved_chunks=chunks_result['chunks']
                    )
                    
                    st.session_state.gdpr_results = gdpr_results
                
                st.success("✅ GDPR Compliance scoring complete!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                st.session_state.is_processing = False
        
        # Display results if available
        if 'gdpr_results' in st.session_state:
            results = st.session_state.gdpr_results
            
            st.markdown("---")
            
            # Overall score card
            col_score, col_grade = st.columns([2, 1])
            
            with col_score:
                st.metric(
                    "Overall GDPR Compliance Score",
                    f"{results['overall_score']}/100",
                    delta=f"Grade: {results['overall_grade']}"
                )
            
            with col_grade:
                # Visual grade indicator
                grade_colors = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴', 'F': '⛔'}
                st.markdown(f"### {grade_colors.get(results['overall_grade'], '⚪')} {results['overall_grade']}")
            
            st.info(results['summary'])
            
            # Strengths and Weaknesses
            if results['strengths'] or results['weaknesses']:
                st.markdown("### 📊 Quick Assessment")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Strengths:**")
                    if results['strengths']:
                        for strength in results['strengths']:
                            st.markdown(f"- {strength}")
                    else:
                        st.markdown("*None identified*")
                
                with col2:
                    st.markdown("**⚠️ Needs Improvement:**")
                    if results['weaknesses']:
                        for weakness in results['weaknesses']:
                            st.markdown(f"- {weakness}")
                    else:
                        st.markdown("*None identified*")
            
            # Detailed principle scores
            st.markdown("---")
            st.markdown("### 📋 Detailed Principle Scores")
            
            for principle in results['principle_scores']:
                with st.expander(f"{principle['grade']} - {principle['principle']} ({principle['score']}/100)"):
                    st.markdown(f"**Description:** {principle['description']}")
                    st.markdown(f"**Assessment:** {principle['assessment']}")
                    
                    # Progress bar
                    st.progress(principle['score'] / 100)
                    
                    # Keywords found
                    if principle['keywords_found']:
                        st.markdown(f"**✓ Found:** {', '.join(principle['keywords_found'])}")
                    
                    # Evidence with source links
                    if principle['evidence']:
                        st.markdown("**📖 Evidence:**")
                        for i, evidence in enumerate(principle['evidence'], 1):
                            # Get source URL if available
                            source_url = st.session_state.get('policy_url', '#')
                            st.markdown(f"{i}. *{evidence[:200]}...* [🔗 View source]({source_url})")
            
            # Recommendations
            if results['recommendations']:
                st.markdown("---")
                st.markdown("### 💡 Recommendations")
                for rec in results['recommendations']:
                    st.markdown(f"- {rec}")
        else:
            st.info("👆 Click 'Score GDPR Compliance' to analyze the policy")
    
    with tab2:
        st.subheader("Comprehensive Privacy Analysis")
        
        btn_disabled = st.session_state.is_processing
        if st.button("🚀 Run Full Analysis", type="primary", disabled=btn_disabled, key="full_analysis_btn"):
            st.session_state.is_processing = True
            policy = st.session_state.policy_text
            url = st.session_state.policy_url
            company = st.session_state.company_name
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.info("🔄 Starting analysis pipeline...")
                progress_bar.progress(10)
                time.sleep(0.3)
                
                status_text.info("📊 Chunking document...")
                progress_bar.progress(20)
                
                status_text.info("🧠 Generating embeddings...")
                progress_bar.progress(40)
                
                status_text.info("🔍 Analyzing with LLM...")
                progress_bar.progress(60)
                
                # Run analysis
                result = st.session_state.pipeline.analyze_policy(
                    text=policy,
                    url=url,
                    company_name=company,
                    dimensions=selected_dims if selected_dims else None,
                    top_k=top_k,
                    llm_client=st.session_state.llm
                )
                
                progress_bar.progress(90)
                status_text.info("📝 Generating report...")
                time.sleep(0.5)
                
                progress_bar.progress(100)
                status_text.success("✅ Analysis complete!")
                
                # Store results
                st.session_state.analysis_results = result
                
                # Display summary
                st.success(f"✅ Analysis complete! Report saved to: {result['report_path']}")
                
                # Show quick preview
                st.subheader("Quick Preview")
                for i, item in enumerate(result['analysis_results'][:3]):
                    with st.expander(f"Q: {item['question']}"):
                        resp = item['response']
                        st.markdown(f"**Answer:** {resp.get('answer', 'N/A')}")
                        st.markdown(f"**Confidence:** `{resp.get('confidence', 'N/A')}`")
                        
                        evidence = resp.get('evidence', [])
                        if evidence:
                            st.markdown("**Evidence:**")
                            for ev in evidence[:2]:
                                st.markdown(f"- [{ev.get('chunk_id')}] {ev.get('quote', '')[:150]}...")
                
            except Exception as e:
                st.error(f"❌ Analysis error: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                st.session_state.is_processing = False
                time.sleep(1.5)
                progress_bar.empty()
                status_text.empty()
                st.rerun()
    
    with tab3:
        st.subheader("Ask Custom Questions")
        
        question = st.text_input("Your Question", placeholder="How is my data protected?", disabled=st.session_state.is_processing)
        
        btn_disabled = st.session_state.is_processing or not question
        if st.button("🔍 Search", key="query_btn", disabled=btn_disabled):
            if question:
                with st.spinner("Retrieving relevant information..."):
                    try:
                        # Index if not already done
                        if st.session_state.pipeline.vector_store.count() == 0:
                            st.session_state.pipeline.index_document(
                                text=st.session_state.policy_text,
                                url=st.session_state.policy_url
                            )
                        
                        # Query
                        result = st.session_state.pipeline.query(
                            question=question,
                            top_k=top_k,
                            return_evidence=True
                        )
                        
                        st.subheader("Retrieved Chunks")
                        
                        for i, chunk in enumerate(result['chunks'][:5], 1):
                            score = chunk.get('hybrid_score', chunk.get('score', 0))
                            with st.expander(f"Chunk {i} (Score: {score:.3f})"):
                                st.markdown(chunk['text'])
                                st.caption(f"Chunk ID: {chunk['chunk_id']}")
                                
                                if 'vector_score' in chunk and 'bm25_score' in chunk:
                                    st.caption(f"Vector: {chunk['vector_score']:.3f} | BM25: {chunk['bm25_score']:.3f}")
                    
                    except Exception as e:
                        st.error(f"Query error: {e}")
            else:
                st.warning("Please enter a question")
    
    with tab4:
        st.subheader("Generated Report")
        
        if 'analysis_results' in st.session_state:
            report_path = st.session_state.analysis_results['report_path']
            
            # Show report
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_md = f.read()
                
                st.markdown(report_md)
                
                # Download button
                st.download_button(
                    label="📥 Download Markdown Report",
                    data=report_md,
                    file_name=os.path.basename(report_path),
                    mime="text/markdown"
                )
                
                # PDF conversion (optional)
                if st.button("📄 Convert to PDF"):
                    with st.spinner("Generating PDF..."):
                        try:
                            st.session_state.pipeline.reporter.convert_to_pdf(report_path)
                            st.success("PDF generated!")
                        except Exception as e:
                            st.warning(f"PDF conversion requires weasyprint: {e}")
            else:
                st.warning("Report file not found")
        else:
            st.info("Run full analysis first to generate a report")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>PrivacyPilot v2.0</strong> - Advanced Privacy Policy Analysis</p>
    <p>Hybrid RAG | Token-Aware Chunking | Evidence-Based</p>
</div>
""", unsafe_allow_html=True)
