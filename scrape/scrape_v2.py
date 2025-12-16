"""
Advanced Privacy Policy Scraper v2.0

Robust multi-page scraping with intelligent content extraction
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import time
import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import concurrent.futures
import re
from typing import Dict, List, Set, Optional


class PrivacyPolicyScraper:
    """
    Intelligent privacy policy scraper with multi-page support
    """
    
    def __init__(self, max_pages: int = 10, timeout: int = 30):
        self.max_pages = max_pages
        self.timeout = timeout
        self.visited: Set[str] = set()
        self.page_contents: Dict[str, Dict] = {}
        self.crawler = None
        
        # Privacy-related URL patterns
        self.privacy_patterns = [
            r'privacy', r'policy', r'policies', r'data.?protection',
            r'gdpr', r'cookie', r'consent', r'terms', r'legal',
            r'compliance', r'security', r'disclosure'
        ]
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL to avoid duplicates"""
        parsed = urlparse(url)
        # Remove fragments and trailing slashes
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path.rstrip('/'),
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        return normalized
    
    def is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain"""
        domain1 = urlparse(url1).netloc.replace('www.', '')
        domain2 = urlparse(url2).netloc.replace('www.', '')
        return domain1 == domain2
    
    def is_privacy_related(self, url: str) -> bool:
        """Check if URL is privacy-related and not just a language variant"""
        url_lower = url.lower()
        
        # Skip language variants (de-de, fr-fr, es-mx, etc.)
        if re.search(r'/[a-z]{2}-[a-z]{2}/', url_lower):
            return False
        
        # Check if any privacy pattern matches
        return any(re.search(pattern, url_lower) for pattern in self.privacy_patterns)
    
    def clean_content(self, html: str, url: str) -> Dict[str, str]:
        """
        Extract clean, relevant content from HTML
        
        Returns:
            {
                'text': cleaned text content,
                'title': page title,
                'headings': list of section headings
            }
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'noscript', 'iframe']):
            element.decompose()
        
        # Extract title
        title = soup.title.string.strip() if soup.title else urlparse(url).path.split('/')[-1]
        
        # Extract headings for structure
        headings = []
        for tag in ['h1', 'h2', 'h3', 'h4']:
            for heading in soup.find_all(tag):
                heading_text = heading.get_text(strip=True)
                if heading_text and len(heading_text) > 2:
                    headings.append(heading_text)
        
        # Try to find main content area
        main_content = None
        
        # Strategy 1: Look for article/main tags
        for tag_name in ['article', 'main']:
            main_content = soup.find(tag_name)
            if main_content:
                break
        
        # Strategy 2: Look for content-related class/id
        if not main_content:
            for pattern in [r'content|main|policy|privacy|article|post', r'container']:
                main_content = soup.find(class_=re.compile(pattern, re.I))
                if main_content:
                    break
                main_content = soup.find(id=re.compile(pattern, re.I))
                if main_content:
                    break
        
        # Strategy 3: Use body
        if not main_content:
            main_content = soup.body or soup
        
        # Remove navigation and other noise AFTER finding main content
        for element in main_content.find_all(['nav', 'header', 'footer', 'aside', 'button', 'form']):
            element.decompose()
        
        # Get text with better formatting
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up text - less aggressive filtering
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            # Keep lines with at least 3 characters (less aggressive than before)
            if len(line) > 3:
                lines.append(line)
        
        clean_text = '\n'.join(lines)
        
        # If still too short, try again with just body text
        if len(clean_text) < 500 and soup.body:
            # Remove only scripts/styles from body
            body_copy = BeautifulSoup(str(soup.body), 'html.parser')
            for element in body_copy(['script', 'style', 'noscript']):
                element.decompose()
            clean_text = body_copy.get_text(separator='\n', strip=True)
        
        return {
            'text': clean_text,
            'title': title,
            'headings': headings[:20]  # Top 20 headings
        }
    
    async def fetch_page_crawl4ai(self, url: str) -> Optional[str]:
        """
        Fetch page using crawl4ai (JavaScript rendering)
        
        Returns HTML content or None
        """
        try:
            if self.crawler is None:
                self.crawler = AsyncWebCrawler(
                    verbose=False,
                    headless=True,
                    browser_type="chromium"
                )
                await self.crawler.start()
            
            print(f"  [OK] Fetching with JS rendering: {url}")
            
            result = await self.crawler.arun(
                url=url,
                wait_for_selector="body",
                delay_before_return=3.0,  # Wait for JS to execute
                js_code=[
                    # Scroll to trigger lazy loading
                    "window.scrollTo(0, document.body.scrollHeight);",
                    # Wait a bit
                    "await new Promise(r => setTimeout(r, 1000));",
                    # Scroll back to top
                    "window.scrollTo(0, 0);"
                ],
                bypass_cache=True,
                timeout=self.timeout
            )
            
            if result and result.html:
                print(f"  [+] Retrieved {len(result.html)} chars of HTML")
                return result.html
            
        except Exception as e:
            print(f"  [!] Crawl4ai error: {str(e)[:100]}")
        
        return None
    
    def fetch_page_requests(self, url: str) -> Optional[str]:
        """
        Fetch page using requests (static HTML)
        
        Returns HTML content or None
        """
        try:
            print(f"  [OK] Fetching static HTML: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            print(f"  [+] Retrieved {len(response.text)} chars of HTML")
            return response.text
            
        except Exception as e:
            print(f"  [!] Requests error: {str(e)[:100]}")
        
        return None
    
    async def scrape_page(self, url: str) -> bool:
        """
        Scrape a single page and store its content
        
        Returns True if successful
        """
        url = self.normalize_url(url)
        
        if url in self.visited:
            return False
        
        self.visited.add(url)
        print(f"\n[*] Scraping ({len(self.visited)}/{self.max_pages}): {url}")
        
        # Try crawl4ai first (for JS-heavy sites)
        html = await self.fetch_page_crawl4ai(url)
        
        # Fallback to requests if crawl4ai fails
        if not html:
            html = self.fetch_page_requests(url)
        
        if not html:
            print(f"  [-] Failed to fetch page")
            return False
        
        # Clean and extract content
        try:
            content = self.clean_content(html, url)
            
            # Validate content quality
            text_length = len(content['text'])
            
            print(f"  [*] Content extraction: {text_length} chars, {len(content['headings'])} headings")
            
            if text_length < 200:
                print(f"  [!] Content too short ({text_length} chars), skipping")
                print(f"      HTML length: {len(html)} chars")
                print(f"      First 200 chars: {html[:200]}")
                return False
            
            # Store content
            self.page_contents[url] = {
                'text': content['text'],
                'title': content['title'],
                'headings': content['headings'],
                'url': url,
                'length': text_length
            }
            
            print(f"  [+] Extracted {text_length:,} chars of clean content")
            print(f"      Title: {content['title'][:60]}")
            print(f"      Sections: {len(content['headings'])}")
            
            return True
            
        except Exception as e:
            print(f"  [-] Content extraction error: {e}")
            return False
    
    def find_related_links(self, url: str, html: str) -> List[str]:
        """
        Find privacy-related links on the page, prioritizing footer links
        
        Returns list of related URLs
        """
        soup = BeautifulSoup(html, 'html.parser')
        related_urls = []
        
        # Priority 1: Footer links (where privacy policies usually are)
        footer_links = []
        for footer in soup.find_all(['footer', 'div'], class_=re.compile(r'footer|bottom|legal', re.I)):
            for link in footer.find_all('a', href=True):
                href = link.get('href')
                link_text = link.get_text(strip=True).lower()
                
                # Check if link text or href contains privacy keywords
                if any(keyword in link_text or keyword in href.lower() 
                       for keyword in self.privacy_patterns):
                    full_url = urljoin(url, href)
                    full_url = self.normalize_url(full_url)
                    
                    if (full_url not in self.visited and
                        self.is_same_domain(url, full_url)):
                        footer_links.append(full_url)
        
        # Priority 2: All other privacy-related links
        body_links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            full_url = urljoin(url, href)
            full_url = self.normalize_url(full_url)
            
            # Check if URL contains privacy keywords
            if (full_url not in self.visited and
                self.is_same_domain(url, full_url) and
                self.is_privacy_related(full_url) and
                full_url not in footer_links):  # Avoid duplicates
                
                body_links.append(full_url)
        
        # Combine: footer links first (higher priority), then body links
        related_urls = footer_links + body_links
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for u in related_urls:
            if u not in seen:
                seen.add(u)
                unique_urls.append(u)
        
        return unique_urls[:15]  # Return top 15 related links
    
    async def scrape_deep(self, start_url: str) -> Dict[str, any]:
        """
        Perform deep scraping starting from a URL
        
        Returns:
            {
                'text': combined text from all pages,
                'title': title of main page,
                'url': start URL,
                'sources': list of scraped sources,
                'page_contents': dict of all page contents
            }
        """
        print(f"\n{'='*70}")
        print(f"[*] Starting Deep Privacy Policy Scrape")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Reset state
        self.visited = set()
        self.page_contents = {}
        
        # BFS queue initialization
        from collections import deque
        queue = deque([start_url])
        
        current_level = 0
        max_levels = 3  # BFS depth limit
        
        try:
            while queue and current_level < max_levels and len(self.page_contents) < self.max_pages:
                level_size = len(queue)
                print(f"\n[+] BFS Level {current_level + 1}: Processing {level_size} URL(s)")
                
                # Process all URLs at current level
                for _ in range(level_size):
                    if len(self.page_contents) >= self.max_pages:
                        break
                    
                    url = queue.popleft()
                    print(f"  [-] Crawling: {url}")
                    
                    # Scrape the page (scrape_page will mark as visited)
                    success = await self.scrape_page(url)
                    
                    if success and current_level < max_levels - 1:
                        # Get HTML for link discovery
                        html = await self.fetch_page_crawl4ai(url)
                        if not html:
                            html = self.fetch_page_requests(url)
                        
                        if html:
                            # Find related links (prioritizes footer)
                            related_links = self.find_related_links(url, html)
                            new_links_count = 0
                            
                            for link in related_links:
                                normalized = self.normalize_url(link)
                                # Don't add to visited yet - let scrape_page handle that
                                if normalized not in self.visited and link not in queue:
                                    queue.append(link)
                                    new_links_count += 1
                            
                            if new_links_count > 0:
                                print(f"  [+] Added {new_links_count} new link(s) to queue")
                    
                    await asyncio.sleep(0.5)  # Rate limiting
                
                current_level += 1
            
            print(f"\n[OK] BFS crawl completed at level {current_level}")
            
            # Compile results
            return self._compile_results(start_url, start_time)
            
        finally:
            # Cleanup
            if self.crawler:
                try:
                    await self.crawler.close()
                except:
                    pass
                self.crawler = None
    
    def _compile_results(self, start_url: str, start_time: float) -> Dict:
        """Compile all scraped content into final result"""
        
        if not self.page_contents:
            return self._create_error_result(start_url, "No content extracted")
        
        # Combine all text with source markers
        combined_text = ""
        sources = []
        
        for url, content in self.page_contents.items():
            # Add source marker
            combined_text += f"\n\n{'='*70}\n"
            combined_text += f"SOURCE: {content['title']}\n"
            combined_text += f"URL: {url}\n"
            combined_text += f"{'='*70}\n\n"
            
            # Add content
            combined_text += content['text']
            
            # Track source
            sources.append({
                'url': url,
                'title': content['title'],
                'length': content['length'],
                'sections': len(content['headings'])
            })
        
        # Get main page info
        main_page = self.page_contents.get(self.normalize_url(start_url))
        if not main_page:
            main_page = list(self.page_contents.values())[0]
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"[SUCCESS] Scraping Complete!")
        print(f"{'='*70}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Pages: {len(self.page_contents)}")
        print(f"  Total: {len(combined_text):,} characters")
        print(f"\nSources:")
        for src in sources:
            print(f"  - {src['title'][:50]}: {src['length']:,} chars")
        print(f"{'='*70}\n")
        
        return {
            'text': combined_text,
            'title': main_page['title'],
            'url': start_url,
            'sources': sources,
            'page_contents': self.page_contents,
            'stats': {
                'pages_scraped': len(self.page_contents),
                'total_chars': len(combined_text),
                'time_seconds': round(elapsed, 1)
            }
        }
    
    def _create_error_result(self, url: str, error_msg: str) -> Dict:
        """Create error result"""
        print(f"[-] {error_msg}")
        return {
            'text': "",
            'title': url.split('/')[2] if '/' in url else url,
            'url': url,
            'sources': [],
            'page_contents': {},
            'error': error_msg
        }


# Global scraper instance
_scraper = None


def scrape(url: str, max_pages: int = 10) -> Dict:
    """
    Main entry point for scraping
    
    Args:
        url: Privacy policy URL to scrape
        max_pages: Maximum number of related pages to scrape
    
    Returns:
        Dictionary with scraped content and metadata
    """
    global _scraper
    
    def _run_async_scrape(url: str) -> Dict:
        """Helper to run async scrape in new event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            scraper = PrivacyPolicyScraper(max_pages=max_pages)
            return loop.run_until_complete(scraper.scrape_deep(url))
        finally:
            loop.close()
    
    # Run in thread pool to avoid event loop conflicts
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_run_async_scrape, url)
        return future.result()


# For testing
if __name__ == "__main__":
    test_url = "https://www.reddit.com/policies/privacy-policy"
    result = scrape(test_url)
    
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Text length: {len(result['text']):,} chars")
    print(f"Pages scraped: {len(result['sources'])}")
    print(f"\nFirst 500 chars:\n{result['text'][:500]}")
