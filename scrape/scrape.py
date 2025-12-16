import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import asyncio
from crawl4ai import AsyncWebCrawler
import concurrent.futures
import threading

from fpdf import FPDF



start_time = time.time()

# Privacy-related keywords for deep crawling
keywords = [
    'privacy', 'policy', 'policies', 'data', 'gdpr', 'protection',
    'cookie', 'cookies', 'terms', 'legal', 'compliance', 'consent',
    'security', 'rights', 'disclosure', 'retention', 'sharing',
    'processing', 'personal', 'information'
]

# Global state
visited = set()
page_contents = {}  # Store content mapped to URLs for source attribution
max_depth = 3  # Crawl up to 3 levels deep
max_pages = 10  # Maximum pages to crawl

# Global crawler instance
crawler = None

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def contains_keyword(url):
    return any(keyword in url.lower() for keyword in keywords)

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def contains_keyword(url):
    return any(keyword in url.lower() for keyword in keywords)

def same_domain(url1, url2):
    """Check if two URLs are from the same domain"""
    domain1 = urlparse(url1).netloc
    domain2 = urlparse(url2).netloc
    return domain1 == domain2


async def scrape_async(url, depth=0, base_domain=None):
    global visited, page_contents, crawler
    
    # Stop if max depth or max pages reached
    if depth > max_depth or len(visited) >= max_pages:
        return
    
    # Skip if already visited
    if url in visited:
        return
    
    visited.add(url)
    
    # Set base domain from first URL
    if base_domain is None:
        base_domain = urlparse(url).netloc
    
    print(f"{'  ' * depth}🔍 Crawling (depth {depth}): {url}")
    
    # Use crawl4ai with JavaScript rendering support
    try:
        if crawler is None:
            crawler = AsyncWebCrawler(
                verbose=False,
                headless=True,
                browser_type="chromium"
            )
            await crawler.start()
        
        # Run with JavaScript execution enabled
        result = await crawler.arun(
            url=url,
            wait_for_selector="body",
            delay_before_return=2.0,
            js_code=["window.scrollTo(0, document.body.scrollHeight);"],
            bypass_cache=True
        )
        
        soup = BeautifulSoup(result.html, 'html.parser')
        
        # Extract text content
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        
        # Store content with source URL
        page_contents[url] = {
            'text': text,
            'title': soup.title.string if soup.title else url,
            'url': url
        }
        
        print(f"{'  ' * depth}✅ Scraped {len(text)} characters from {url}")
        
        # Find and follow related links (only if not at max depth)
        if depth < max_depth and len(visited) < max_pages:
            links_to_follow = []
            
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                
                # Only follow links in same domain and containing privacy keywords
                if (same_domain(full_url, url) and 
                    contains_keyword(full_url) and 
                    full_url not in visited and
                    is_valid(full_url)):
                    links_to_follow.append(full_url)
            
            # Follow up to 3 related links per page
            for related_url in links_to_follow[:3]:
                await scrape_async(related_url, depth + 1, base_domain)
        
    except Exception as e:
        print(f"{'  ' * depth}❌ Error scraping {url}: {e}")
        # Try requests fallback
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            page_contents[url] = {
                'text': text,
                'title': soup.title.string if soup.title else url,
                'url': url
            }
            print(f"{'  ' * depth}✅ Fallback scraped {len(text)} characters")
        except Exception as fallback_error:
            print(f"{'  ' * depth}❌ Fallback also failed: {fallback_error}")


async def scrape_and_cleanup(url):
    """Async wrapper that handles both scraping and cleanup"""
    global crawler, visited, page_contents
    try:
        # Scrape main URL and related pages
        await scrape_async(url, depth=0)
        
        # Combine all scraped content
        all_text = ""
        sources = []
        
        for source_url, content in page_contents.items():
            all_text += f"\n\n--- SOURCE: {source_url} ---\n\n"
            all_text += content['text']
            sources.append({
                'url': source_url,
                'title': content['title'],
                'length': len(content['text'])
            })
        
        # Extract title from first page
        first_page = list(page_contents.values())[0] if page_contents else None
        title = first_page['title'] if first_page else url.split('/')[2]
        
        print(f"\n📊 Scraping Summary:")
        print(f"  - Pages scraped: {len(page_contents)}")
        print(f"  - Total content: {len(all_text):,} characters")
        for src in sources:
            print(f"  - {src['title'][:50]}: {src['length']:,} chars")
        
        # Return dictionary format expected by the app with source mapping
        return {
            'text': all_text or "",
            'title': title,
            'url': url,
            'sources': sources,
            'page_contents': page_contents  # For citation/reference
        }
    finally:
        # Always cleanup the crawler
        if crawler is not None:
            try:
                await crawler.close()
            except Exception:
                pass
            crawler = None


def scrape(url):
    """
    Wrapper function to run async scrape function.
    Returns a dictionary with 'text', 'title', 'url', 'sources', and 'page_contents' keys.
    """
    # Reset global state for each new scrape
    global visited, page_contents, start_time
    visited = set()
    page_contents = {}
    start_time = time.time()
    
    print(f"🚀 Starting deep crawl for: {url}")
    
    # Always use thread pool to avoid event loop conflicts with Streamlit
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_run_async_scrape, url)
        result = future.result()
    
    print(f"✅ Deep crawl complete!")
    return result

def _run_async_scrape(url):
    """Helper to run async scrape in a new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(scrape_and_cleanup(url))
    finally:
        loop.close()


from fpdf import FPDF

def save_to_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()

    font_path = r"scrape\DejaVuSans.ttf"
    try:
        pdf.add_font('DejaVu', '', font_path, uni=True)
        pdf.set_font('DejaVu', size=12)
    except RuntimeError as e:
        print(f"Error loading font: {e}")
        return

    count = 0
    for line in text.split('\n'):
        try:
            safe_line = ''.join(c if ord(c) < 65536 else '?' for c in line)
            pdf.multi_cell(0, 10, safe_line)
            count+=1
        except Exception as e:
            print(f"Error writing line: {line}\n{e}")

    try:
        pdf.output(filename)
        print(f"PDF saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving PDF: {e}")


