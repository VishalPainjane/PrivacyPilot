import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import requests

def get_first_google_result(url):
    # Extract domain name from URL (e.g., "reddit" from "https://www.reddit.com/...")
    match = re.search(r'https?://(?:www\.)?([^./]+)', url)
    if match:
        context = match.group(1)
    else:
        # Fallback: try to get domain from the full URL
        context = url.replace('https://', '').replace('http://', '').split('/')[0].replace('www.', '')

    query = f"{context} terms and conditions"
    print(f"Searching Google for: {query}")
    encoded_query = quote_plus(query)
    google_url = f"https://www.google.com/search?q={encoded_query}"

    # Use requests for Google search (more reliable than crawl4ai for simple searches)
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(google_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the first search result link
        first_link = soup.select_one('div.yuRUbf a')
        if first_link and first_link.get('href'):
            print(f"Found Google result: {first_link.get('href')}")
            return first_link.get('href')
        
        # Try alternative selectors
        links = soup.select('a[href^="http"]')
        for link in links:
            href = link.get('href')
            if href and 'google.com' not in href and 'youtube.com' not in href and 'gstatic.com' not in href:
                print(f"Found Google result (alternative): {href}")
                return href
    except Exception as e:
        print(f"Error searching Google: {e}")
    
    # Return original URL if nothing found
    print(f"Using original URL: {url}")
    return url
