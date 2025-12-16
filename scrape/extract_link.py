import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import requests
import time

def get_first_google_result(query):
    """
    Search for privacy policy URL using multiple strategies.
    
    Args:
        query: Search query (e.g., "reddit privacy policy" or company name)
    
    Returns:
        Privacy policy URL or None if nothing found
    """
    print(f"Searching for: {query}")
    
    # Common company to domain mappings
    DOMAIN_MAPPINGS = {
        'meta': 'facebook.com',
        'facebook': 'facebook.com',
        'google': 'google.com',
        'microsoft': 'microsoft.com',
        'amazon': 'amazon.com',
        'apple': 'apple.com',
        'netflix': 'netflix.com',
        'twitter': 'twitter.com',
        'x': 'twitter.com',
        'linkedin': 'linkedin.com',
        'reddit': 'reddit.com',
        'instagram': 'instagram.com',
        'whatsapp': 'whatsapp.com',
        'youtube': 'youtube.com',
        'tiktok': 'tiktok.com',
        'snapchat': 'snap.com',
        'discord': 'discord.com',
        'slack': 'slack.com',
        'zoom': 'zoom.us',
        'spotify': 'spotify.com',
        'dropbox': 'dropbox.com',
        'adobe': 'adobe.com',
        'salesforce': 'salesforce.com',
        'oracle': 'oracle.com',
        'ibm': 'ibm.com',
        'nvidia': 'nvidia.com',
        'uber': 'uber.com',
        'airbnb': 'airbnb.com',
        'paypal': 'paypal.com',
        'ebay': 'ebay.com',
        'walmart': 'walmart.com',
        'target': 'target.com',
        'overleaf': 'overleaf.com',
    }
    
    # Strategy 1: Direct URL construction
    domain = None
    if query.startswith('http://') or query.startswith('https://'):
        match = re.search(r'https?://(?:www\.)?([^./]+\.[^./]+)', query)
        if match:
            domain = match.group(1)
    else:
        # Extract company name and map to domain
        company = re.sub(r'\s*(privacy|policy|data|protection).*', '', query.lower().strip())
        company = company.replace(' ', '').replace('.com', '').replace('.org', '').replace('.net', '')
        
        # Try mapping first
        if company in DOMAIN_MAPPINGS:
            domain = DOMAIN_MAPPINGS[company]
            print(f"  Mapped '{company}' → {domain}")
        elif '.' not in company:
            # Assume .com for unknown companies
            domain = f"{company}.com"
        else:
            domain = company
    
    # Try common privacy policy URL patterns
    if domain:
        common_patterns = [
            f"https://www.{domain}/privacy",
            f"https://{domain}/privacy",
            f"https://www.{domain}/legal",
            f"https://{domain}/legal",
            f"https://www.{domain}/privacy-policy",
            f"https://{domain}/privacy-policy",
            f"https://www.{domain}/legal/privacy",
            f"https://{domain}/legal/privacy",
            f"https://policies.{domain}/privacy",
            f"https://{domain}/policies/privacy",
            f"https://{domain}/en/privacy",
            f"https://{domain}/about/privacy",
            f"https://{domain}/legal/privacy-policy",
            f"https://help.{domain}/privacy",
            f"https://www.{domain}/terms/privacy",
        ]
        
        for url in common_patterns:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
                if response.status_code == 200:
                    final_url = response.url if response.url != url else url
                    print(f"✅ Found via direct URL: {final_url}")
                    return final_url
            except:
                continue
    
    # Strategy 2: Search engines as fallback
    search_query = query + " privacy policy" if "privacy" not in query.lower() else query
    encoded_query = quote_plus(search_query)
    
    # Try multiple search engines for better reliability
    search_engines = [
        {
            'name': 'Google',
            'url': f"https://www.google.com/search?q={encoded_query}",
            'selectors': ['div.yuRUbf a', 'div.g a[href^="http"]', 'a[href^="http"]']
        },
        {
            'name': 'DuckDuckGo',
            'url': f"https://html.duckduckgo.com/html/?q={encoded_query}",
            'selectors': ['a.result__a']
        }
    ]

    for engine in search_engines:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(engine['url'], headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try each selector
            for selector in engine['selectors']:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    
                    # Clean DuckDuckGo redirect URLs
                    if 'uddg=' in href:
                        href = re.search(r'uddg=([^&]+)', href)
                        if href:
                            from urllib.parse import unquote
                            href = unquote(href.group(1))
                    
                    # Filter out unwanted URLs
                    if href and all(x not in href.lower() for x in [
                        'google.com', 'youtube.com', 'gstatic.com', 
                        'googleusercontent.com', 'duckduckgo.com'
                    ]):
                        # Make sure it's a full URL
                        if href.startswith('http'):
                            print(f"✅ Found via {engine['name']}: {href}")
                            return href
                            
        except Exception as e:
            print(f"⚠️ {engine['name']} search error: {e}")
            continue
    
    print(f"⚠️ No results found via any search engine for: {query}")
    return None
