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

# keywords = ['privacy', 'terms', 'condition', 'legal']

keywords = [
    'term', 'terms', 'condition', 'conditions', 'policy', 'policies',
    'agreement', 'agreements', 'statement', 'statements',
    'protection', 'protections', 'clause', 'clauses', 'disclaimer', 'disclaimers', 'legal',
    'privacy', 'data', 'service', 'user', 'cookie', 'third-party', 'security', 'consent',
    'notice', 'regulation', 'compliance', 'contract', 'contracts', 'rights', 'sharing',
    'opt-out', 'opt-in', 'disclosure', 'retention', 'encryption', 'Terms of service', 'safety',
    'help', 'support', 'contact', 'information', 'cookies', 'cookies-policy', 
    'user agreement', 'cookie policy', 'terms of service', 'privacy policy', 'privacy-policy'
    'privacy statement', 'data protection', 'legal terms', 'legal agreement',
    'service terms', 'service agreement', 'terms and conditions', 't&c', 't and c',
    't n c', 't&cs', 't&cs policy', 'tnc policy', 'tncs','Privacy policy',

    'regulation', 'compliance', 'contract', 'contracts', 'user consent',
    'data usage', 'data collection', 'user rights', 'third-party sharing',
    'opt-out', 'opt-in', 'disclosure', 'retention', 'security policy',
]


visited = set()
extracted_text = ""

# Global crawler instance
crawler = None

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def contains_keyword(url):
    return any(keyword in url.lower() for keyword in keywords)

count = 0
flag = 0
start_link = ""


async def scrape_async(url):
    global start_time
    global extracted_text
    global count
    global start_link
    global flag
    global crawler

    if count == 0:
        start_link = url
        count += 1

    if time.time() - start_time > 7:
        return extracted_text[:200000] or ""

    if url in visited:
        return

    visited.add(url)
    
    # Try crawl4ai first, fallback to requests if it fails
    try:
        if crawler is None:
            crawler = AsyncWebCrawler(verbose=False)
            await crawler.start()
        
        result = await crawler.arun(url=url)
        soup = BeautifulSoup(result.html, 'html.parser')
        text = soup.get_text()
    except Exception as crawl_error:
        # Fallback to requests if crawl4ai fails
        print(f"Crawl4ai error, using requests fallback: {crawl_error}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
        except Exception as req_error:
            print(f"Requests fallback also failed: {req_error}")
            return extracted_text[:200000] or ""
    
    try:
        if extracted_text is None:
            extracted_text = ""
        extracted_text += text

        for link in soup.find_all('a', href=True):
            if time.time() - start_time > 7:
                return extracted_text[:200000] or ""

            domain = start_link[len("https://"):].split(".")[0]
            if domain not in link['href']:
                continue

            full_url = urljoin(url, link['href'])
            if flag != 0:
                if full_url in visited:
                    continue

            if is_valid(full_url) and contains_keyword(full_url):
                result = await scrape_async(full_url)
                flag = 1
                if result:
                    extracted_text += result
    except Exception as e:
        print(f"Error processing scraped content: {e}")

    return extracted_text[:200000] or ""


async def scrape_and_cleanup(url):
    """Async wrapper that handles both scraping and cleanup"""
    global crawler
    try:
        result = await scrape_async(url)
        return result
    finally:
        # Always cleanup the crawler
        if crawler is not None:
            try:
                await crawler.close()
            except Exception:
                pass  # Ignore cleanup errors
            crawler = None


def scrape(url):
    """Wrapper function to run async scrape function"""
    # Always use thread pool to avoid event loop conflicts with Streamlit
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_run_async_scrape, url)
        return future.result()

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


